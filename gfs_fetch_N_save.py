import pygrib
import numpy as np
import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from config import get_config
# generator 모듈에서 함수 불러오기
from gfs_gen_image import process_and_save_image_com

# =================================================================
# 1. 전역 설정
# =================================================================
DOWNSAMPLE = False
SLEEP_TIME = 3600
DATA_RES = "0p25"
MAX_TIME_OFFSET = 120 # 최대 시간 오프셋 (예: 120시간까지)
REMOVE_JSON_AFTER_PROCESS = True # 처리 후 JSON 파일 삭제 여부

EXTRACTION_CONFIG = {
    "TMP": {
        "var_key": "var_TMP",
        "levels": ["lev_2_m_above_ground", "lev_850_mb", "lev_500_mb"],
        "params": {
            "lev_2_m_above_ground": {"name": "2 metre temperature", "type": "heightAboveGround", "level": 2, "suffix": "2m"},
            "lev_850_mb": {"name": "Temperature", "type": "isobaricInhPa", "level": 850, "suffix": "850mb"},
            "lev_500_mb": {"name": "Temperature", "type": "isobaricInhPa", "level": 500, "suffix": "500mb"},
        }
    },
    "RH": {
        "var_key": "var_RH",
        "levels": ["lev_2_m_above_ground", "lev_850_mb", "lev_500_mb"],
        "params": {
            "lev_2_m_above_ground": {"name": "2 metre relative humidity", "type": "heightAboveGround", "level": 2, "suffix": "2m"},
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

    print(f"checking existing file for GFS {final_path}")
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
            os.rename(temp_path, final_path); 
            return final_path, True # 새로 받은 파일 여부 반환
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

# =================================================================
# 3. 메인 프로세스
# =================================================================

def run_fetcher():
    while True:
        date, hour = get_utc_date_time_com()
        
        for offset in TIME_OFFSETS:
            base_utc_dt = datetime.strptime(f"{date}{hour}00", '%Y%m%d%H%M')
            utc_dt = base_utc_dt + timedelta(hours=int(offset))
            kor_dt = utc_dt + timedelta(hours=9)
            
            utc_str = utc_dt.strftime('%Y%m%d%H%M')
            kor_str = kor_dt.strftime('%Y%m%d%H%M')
            sub_dir = kor_dt.strftime('%Y-%m-%d')
            os.makedirs(f"{OUT_DIR}/{sub_dir}", exist_ok=True)

            grib_file, is_new = download_combined_gfs_com(date, hour, offset)
            # 만약 새로 받은 파일이 아니라면(이미 처리된 사이클), 다음 offset으로 스킵
            if not is_new:
                # 단, 만약 raw 파일은 있는데 결과물(png/json)이 사고로 삭제되었을 경우를 대비해 
                # 체크 로직을 넣고 싶다면 여기서 체크 가능합니다.
                continue
            print(f"   => New data found! Processing F{offset}...")

            try:
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
                            # 2. 모듈 호출하여 PNG 저장
                            process_and_save_image_com(results, t_key, f"{file_base}.png")
                            # print(f"   [SUCCESS] Saved: gfs_{t_key.lower()}_{p['suffix']}...")
                            print(f"   [SUCCESS] Saved json&png: {file_base}")
                            if REMOVE_JSON_AFTER_PROCESS and os.path.exists(f"{file_base}.json"):
                                os.remove(f"{file_base}.json")

                grbs.close()
            except Exception as e:
                print(f"   [ERROR] Processing Loop: {e}")

        print(f"Cycle finished. Sleeping {SLEEP_TIME}s..."); time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    run_fetcher()