import pygrib
import numpy as np
import json
import os
import time
import requests
import argparse
import re
from datetime import datetime, timezone, timedelta
from glob import glob
from config import get_config
# generator 모듈에서 함수 불러오기
from gfs_gen_image import process_and_save_image_com

# =================================================================
# 1. 전역 설정
# =================================================================
DOWNSAMPLE = False
SLEEP_TIME = 3600
DATA_RES = "0p25"
MAX_TIME_OFFSET = 120 
REMOVE_JSON_AFTER_PROCESS = True 

EXTRACTION_CONFIG = {
    "TMP": {
        "var_key": "var_TMP",
        "levels": ["lev_2_m_above_ground", "lev_850_mb", "lev_500_mb"],
        "params": {
            "lev_2_m_above_ground": {"name": "2 metre temperature", "type": "heightAboveGround", "level": 2, "suffix": "10m"},
            "lev_850_mb": {"name": "Temperature", "type": "isobaricInhPa", "level": 850, "suffix": "850mb"},
            "lev_500_mb": {"name": "Temperature", "type": "isobaricInhPa", "level": 500, "suffix": "500mb"},
        }
    },
    "RH": {
        "var_key": "var_RH",
        "levels": ["lev_2_m_above_ground", "lev_850_mb", "lev_500_mb"],
        "params": {
            "lev_2_m_above_ground": {"name": "2 metre relative humidity", "type": "heightAboveGround", "level": 2, "suffix": "10m"},
            "lev_850_mb": {"name": "Relative humidity", "type": "isobaricInhPa", "level": 850, "suffix": "850mb"},
            "lev_500_mb": {"name": "Relative humidity", "type": "isobaricInhPa", "level": 500, "suffix": "500mb"},
        }
    },
    "WIND": {
        "var_key": ["var_UGRD", "var_VGRD"],
        "levels": ["lev_10_m_above_ground", "lev_850_mb", "lev_500_mb"],
        "params": {
            "lev_10_m_above_ground": {"U_name": "10 metre U wind component", "V_name": "10 metre V wind component", "type": "heightAboveGround", "level": 10, "suffix": "10m"},
            "lev_850_mb": {"U_name": "U component of wind", "V_name": "V component of wind", "type": "isobaricInhPa", "level": 850, "suffix": "850mb"},
            "lev_500_mb": {"U_name": "U component of wind", "V_name": "V component of wind", "type": "isobaricInhPa", "level": 500, "suffix": "500mb"},
        }
    }
}

TIME_OFFSETS = [f"{i:03d}" for i in range(MAX_TIME_OFFSET + 1)]
config = get_config()
IN_DIR, OUT_DIR = config.WATCH_PATH_WIND, config.OUT_PATH_WIND

# =================================================================
# 2. 공용 보조 함수 (_com)
# =================================================================

def get_utc_date_time_com():
    now = datetime.now(timezone.utc)
    valid_hours = [0, 6, 12, 18]
    past_hour = max([h for h in valid_hours if h <= now.hour], default=18)
    return now.strftime("%Y%m%d"), f"{past_hour:02d}"

def download_combined_gfs_com(date, hour, offset):
    base_name = f"GFS_{date}_{hour}_{offset}"
    final_path = f"{IN_DIR}/combined_raw/{base_name}_done.grib2"
    temp_path = f"{IN_DIR}/combined_raw/{base_name}.tmp"

    if os.path.exists(final_path): 
        print(f"[FETCH] Using existing GFS {date} {hour}Z, F{offset}...")
        return final_path, False
    
    os.makedirs(f"{IN_DIR}/combined_raw", exist_ok=True)
    params = {"dir": f"/gfs.{date}/{hour}/atmos/", "file": f"gfs.t{hour}z.pgrb2.{DATA_RES}.f{offset}"}
    for cfg in EXTRACTION_CONFIG.values():
        v_keys = cfg["var_key"] if isinstance(cfg["var_key"], list) else [cfg["var_key"]]
        for vk in v_keys: params[vk] = "on"
        for lvl in cfg["levels"]: params[lvl] = "on"
    
    url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{DATA_RES}.pl"
    try:
        print(f"[FETCH] Requesting GFS {date} {hour}Z, F{offset}...")
        res = requests.get(url, params=params, timeout=60, stream=True)
        if res.status_code == 200:
            with open(temp_path, 'wb') as f:
                for chunk in res.iter_content(8192): f.write(chunk)
            os.rename(temp_path, final_path)
            return final_path, True
        else:
            print(f"   => Download failed with status code: {res.status_code}")
    except Exception as e:
        print(f"   => Download Error: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
    return None, False

def format_header_com(name, res):
    return {
        "parameterName": name, "nx": 1440, "ny": 721,
        "lo1": 0.0, "la1": 90.0, "lo2": 359.75, "la2": -90.0,
        "dx": res, "dy": res
    }

def process_grib_file_com(grib_file, date, hour, offset, remove_json=REMOVE_JSON_AFTER_PROCESS):
    """GRIB 파일 하나를 처리하여 JSON/PNG 생성"""
    try:
        base_utc_dt = datetime.strptime(f"{date}{hour}00", '%Y%m%d%H%M')
        utc_dt = base_utc_dt + timedelta(hours=int(offset))
        kor_dt = utc_dt + timedelta(hours=9)
        
        utc_str = utc_dt.strftime('%Y%m%d%H%M')
        kor_str = kor_dt.strftime('%Y%m%d%H%M')
        sub_dir = kor_dt.strftime('%Y-%m-%d')
        os.makedirs(f"{OUT_DIR}/{sub_dir}", exist_ok=True)

        grbs = pygrib.open(grib_file)
        for t_key, t_cfg in EXTRACTION_CONFIG.items():
            for lvl_key in t_cfg["levels"]:
                p = t_cfg["params"][lvl_key]
                grbs.seek(0)
                results = []

                if t_key == "WIND":
                    u = v = None
                    for g in grbs:
                        if g.level == p["level"] and g.typeOfLevel == p["type"]:
                            if g.name == p["U_name"]: u = g.values.tolist()
                            if g.name == p["V_name"]: v = g.values.tolist()
                    if u and v:
                        results = [
                            {"header": format_header_com(p["U_name"], 0.25), "data": u},
                            {"header": format_header_com(p["V_name"], 0.25), "data": v}
                        ]
                else:
                    for g in grbs:
                        if g.name == p["name"] and g.level == p["level"] and g.typeOfLevel == p["type"]:
                            results = [{"header": format_header_com(p["name"], 0.25), "data": g.values.tolist()}]
                            break
                
                if results:
                    file_base = f"{OUT_DIR}/{sub_dir}/gfs_{DATA_RES}_{t_key.lower()}_{p['suffix']}_{utc_str}_{kor_str}"
                    # 1. JSON 저장
                    with open(f"{file_base}.json", 'w') as f:
                        json.dump(results, f)
                    # 2. PNG 저장
                    process_and_save_image_com(results, t_key, f"{file_base}.png")
                    print(f"   [SUCCESS] Saved json&png: {file_base}")
                    
                    if remove_json and os.path.exists(f"{file_base}.json"):
                        os.remove(f"{file_base}.json")
        grbs.close()
    except Exception as e:
        print(f"   [ERROR] Processing {grib_file}: {e}")

# =================================================================
# 3. 실행 모드 루틴
# =================================================================

def run_fetcher():
    """기존 실시간 무한 루프 모드"""
    print("Starting Fetcher Mode...")
    while True:
        date, hour = get_utc_date_time_com()
        for offset in TIME_OFFSETS:
            grib_file, is_new = download_combined_gfs_com(date, hour, offset)
            if is_new or grib_file: # 여기서는 is_new 체크를 통해 중복 방지
                if is_new:
                    print(f"   => New data found! Processing F{offset}...")
                    process_grib_file_com(grib_file, date, hour, offset)
            
        print(f"Cycle finished. Sleeping {SLEEP_TIME}s..."); time.sleep(SLEEP_TIME)

def run_batch_reprocess(target_dir):
    """특정 폴더의 GRIB2 파일을 모두 읽어 재처리"""
    print(f"Starting Batch Reprocess Mode: {target_dir}")
    # GFS_YYYYMMDD_HH_FFF_done.grib2 형태 검색
    files = glob(os.path.join(target_dir, "*.grib2"))
    
    if not files:
        print("No .grib2 files found in the directory.")
        return

    for grib_file in sorted(files):
        # 파일명에서 정보 추출 (패턴: GFS_20260311_00_000_done.grib2)
        match = re.search(r'GFS_(\d{8})_(\d{2})_(\d{3})', os.path.basename(grib_file))
        if match:
            date, hour, offset = match.groups()
            print(f"Processing batch file: {os.path.basename(grib_file)}")
            remove_json_file = False # 배치 모드에서는 JSON 파일을 유지할지 여부 (필요시 True로 변경)
            process_grib_file_com(grib_file, date, hour, offset, remove_json_file)
        else:
            print(f"Skipping file (name format mismatch): {os.path.basename(grib_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFS Fetch and Process Tool")
    parser.add_argument("--grib-dir", type=str, help="Directory path to reprocess all GRIB2 files")
    
    args = parser.parse_args()
    
    if args.grib_dir:
        # 재생성 모드 실행
        run_batch_reprocess(args.grib_dir)
    else:
        # 기존 실시간 루프 모드 실행
        run_fetcher()