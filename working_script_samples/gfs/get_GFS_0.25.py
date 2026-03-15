import pygrib
import numpy as np
import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from config import get_config

# =================================================================
# 1. 전역 설정 (모든 수정 및 확장은 여기서 관리합니다)
# =================================================================
DOWNSAMPLE = False  # True: 1.0도 변환, False: 0.25도 유지
SLEEP_TIME = 600    # 메인 루프 대기 시간 (초)
DATA_RES = "0p25"   # GFS 데이터 해상도

# 추출 설정: 데이터 타입별 NOAA 필터 변수명, 레벨, 세부 파라미터 정의
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
            "lev_10_m_above_ground": {
                "U_name": "10 metre U wind component", "V_name": "10 metre V wind component",
                "type": "heightAboveGround", "level": 10, "suffix": "10m"
            },
            "lev_850_mb": {
                "U_name": "U component of wind", "V_name": "V component of wind",
                "type": "isobaricInhPa", "level": 850, "suffix": "850mb"
            },
            "lev_500_mb": {
                "U_name": "U component of wind", "V_name": "V component of wind",
                "type": "isobaricInhPa", "level": 500, "suffix": "500mb"
            },
        }
    },
    "PRMSL": {
        "var_key": "var_PRMSL",
        "levels": ["lev_mean_sea_level"],
        "params": {
            "lev_mean_sea_level": {"name": "Pressure reduced to MSL", "type": "meanSeaLevel", "level": 0, "suffix": "msl"}
        }
    }
}

# 예측 시간 오프셋 (000 ~ 120)
TIME_OFFSETS = [f"{i:03d}" for i in range(121)]

# 경로 설정
config = get_config()
IN_DIR = config.WATCH_PATH_WIND
OUT_DIR = config.OUT_PATH_WIND

# =================================================================
# 2. 공용 함수 (_com)
# =================================================================

def get_utc_date_time_com():
    """현재 기준 가장 최신 GFS 사이클 시간 반환"""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    valid_hours = [0, 6, 12, 18]
    past_hour = max([h for h in valid_hours if h <= now.hour], default=18)
    return date_str, f"{past_hour:02d}"

def add_hours_com(dt_str, hours):
    """문자열 형태의 시간에 오프셋 더하기"""
    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    dt = dt + timedelta(hours=hours)
    return dt.strftime('%Y%m%d%H%M')

def download_combined_gfs_com(date, hour, offset):
    """설정을 기반으로 필요한 모든 변수를 한 번에 다운로드"""
    base_name = f"GFS_{date}_{hour}_{offset}"
    final_path = f"{IN_DIR}/combined_raw/{base_name}_done.grib2"
    temp_path = f"{IN_DIR}/combined_raw/{base_name}.tmp"

    # 중복 체크: 이미 완료된 파일이 있으면 스킵
    if os.path.exists(final_path):
        print(f"[FETCH] Using existing GFS {date} {hour}Z, F{offset}...")
        return final_path

    os.makedirs(f"{IN_DIR}/combined_raw", exist_ok=True)
    
    # 동적 파라미터 구성
    params = {
        "dir": f"/gfs.{date}/{hour}/atmos/",
        "file": f"gfs.t{hour}z.pgrb2.{DATA_RES}.f{offset}",
    }
    for cfg in EXTRACTION_CONFIG.values():
        v_keys = cfg["var_key"] if isinstance(cfg["var_key"], list) else [cfg["var_key"]]
        for vk in v_keys: params[vk] = "on"
        for lvl in cfg["levels"]: params[lvl] = "on"
    
    url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{DATA_RES}.pl"
    
    try:
        print(f"[FETCH] Requesting GFS {date} {hour}Z, F{offset}...")
        response = requests.get(url, params=params, timeout=60, stream=True)
        if response.status_code == 200:
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            os.rename(temp_path, final_path)
            return final_path
        else:
            print(f"   => Fail: HTTP {response.status_code}")
    except Exception as e:
        print(f"   => Download Error: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
    return None

def downsample_com(data):
    """(721, 1440) -> (181, 360) 다운샘플링"""
    new_lat, new_lon = 181, 360
    res_data = np.zeros((new_lat, new_lon))
    for i in range(new_lat - 1):
        for j in range(new_lon):
            res_data[i, j] = np.mean(data[i*4:i*4+4, j*4:j*4+4])
    for j in range(new_lon):
        res_data[-1, j] = np.mean(data[-1, j*4:j*4+4])
    return np.round(res_data, 2)

def format_json_data_com(name, data, res):
    """표준화된 JSON 헤더 생성"""
    ny, nx = data.shape
    return {
        "header": {
            "parameterName": name,
            "nx": nx, "ny": ny,
            "lo1": 0.0, "la1": 90.0,
            "lo2": 359.75 if res == 0.25 else 359.0,
            "la2": -90.0,
            "dx": res, "dy": res
        },
        "data": data.flatten().tolist()
    }

# =================================================================
# 3. 데이터 처리 로직
# =================================================================

def extract_and_save_all_com(file_path, utc_str, kor_str):
    """GRIB 파일에서 설정된 모든 기상 요소 추출 및 저장"""
    try:
        grbs = pygrib.open(file_path)
    except Exception as e:
        print(f"   [ERROR] GRIB Open Fail: {e}")
        return

    res_val = 0.25 if not DOWNSAMPLE else 1.0
    res_prefix = "025" if not DOWNSAMPLE else "100"
    sub_dir = f"{kor_str[:4]}-{kor_str[4:6]}-{kor_str[6:8]}"
    os.makedirs(f"{OUT_DIR}/{sub_dir}", exist_ok=True)

    for type_key, t_cfg in EXTRACTION_CONFIG.items():
        for lvl_key in t_cfg["levels"]:
            p = t_cfg["params"][lvl_key]
            grbs.seek(0)
            extracted_results = []
            
            if type_key == "WIND":
                u_raw = v_raw = None
                for grb in grbs:
                    if grb.typeOfLevel == p["type"] and grb.level == p["level"]:
                        if grb.name == p["U_name"]: u_raw = grb.values
                        if grb.name == p["V_name"]: v_raw = grb.values
                if u_raw is not None and v_raw is not None:
                    u_f = downsample_com(u_raw) if DOWNSAMPLE else np.round(u_raw, 2)
                    v_f = downsample_com(v_raw) if DOWNSAMPLE else np.round(v_raw, 2)
                    extracted_results.append(format_json_data_com(p["U_name"], u_f, res_val))
                    extracted_results.append(format_json_data_com(p["V_name"], v_f, res_val))
            else:
                val_raw = None
                for grb in grbs:
                    if grb.name == p["name"] and grb.typeOfLevel == p["type"] and grb.level == p["level"]:
                        val_raw = grb.values
                        break
                if val_raw is not None:
                    val_f = downsample_com(val_raw) if DOWNSAMPLE else np.round(val_raw, 2)
                    extracted_results.append(format_json_data_com(p["name"], val_f, res_val))

            if extracted_results:
                fname = f"gfs_{type_key.lower()}_{p['suffix']}_{res_prefix}_{utc_str}_{kor_str}.json"
                with open(f"{OUT_DIR}/{sub_dir}/{fname}", 'w') as f:
                    json.dump(extracted_results, f)
                print(f"   [SUCCESS] Saved: {fname}")
    
    grbs.close()

# =================================================================
# 4. 메인 실행 루프
# =================================================================

if __name__ == "__main__":
    while True:
        date, hour = get_utc_date_time_com()
        
        for offset in TIME_OFFSETS:
            base_utc = f"{date}{hour}00"
            utc_str = add_hours_com(base_utc, int(offset))
            kor_str = add_hours_com(utc_str, 9)
            
            # 다운로드 시도 (이미 완료된 파일이 있으면 함수 내부에서 즉시 반환)
            grib_file = download_combined_gfs_com(date, hour, offset)
            
            if grib_file:
                try:
                    # 파일이 확보되면 JSON 추출 및 Overwrite 수행
                    extract_and_save_all_com(grib_file, utc_str, kor_str)
                except Exception as e:
                    print(f"   [ERROR] Main Process Error: {e}")

        print(f"Cycle finished at {datetime.now()}. Waiting {SLEEP_TIME}s...")
        time.sleep(SLEEP_TIME)