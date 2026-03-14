import pygrib
import numpy as np
import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from config import get_config

# ==========================================
# 설정: 0.25도 데이터를 그대로 사용할지 여부
DOWNSAMPLE = False 
# ==========================================

config = get_config()

# 추출할 기상 레벨 정의 (상대습도 RH 전용)
# GFS에서 RH는 2m, 850mb, 500mb 등에서 제공됩니다.
LEVELS = ["lev_2_m_above_ground", "lev_850_mb", "lev_500_mb"]
PARAMS_FOR_LEVELS = {
    "lev_2_m_above_ground": {
        "RH_name": "2 metre relative humidity", # GFS 공식 명칭
        "typeOfLevel": "heightAboveGround",
        "level": 2,
        "fname_level": "2m"
    },
    "lev_500_mb": {
        "RH_name": "Relative humidity",
        "typeOfLevel": "isobaricInhPa",
        "level": 500,
        "fname_level": "500mb"
    },
    "lev_850_mb": {
        "RH_name": "Relative humidity",
        "typeOfLevel": "isobaricInhPa",
        "level": 850,
        "fname_level": "850mb"
    },
}

VALUES = ["on"] * len(LEVELS)
in_dir = config.WATCH_PATH_WIND
out_dir = config.OUT_PATH_WIND

def get_utc_date_time():
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    valid_hours = [0, 6, 12, 18]
    current_hour = now.hour
    past_hour = max([h for h in valid_hours if h <= current_hour], default=18)
    return date_str, f"{past_hour:02d}"

def download_file(date, hours, in_dir, sub_dir, level="0p25", time_offset="000"):
    date = '20260311'
    hours = '00' # 테스트용 고정 (실사용 시 get_utc_date_time의 hour 사용)
    print(f"Downloading GFS RH data for {date} {hours}Z, F{time_offset}")
    
    params = {
        "dir": f"/gfs.{date}/{hours}/atmos/",
        "file": f"gfs.t{hours}z.pgrb2.{level}.f{time_offset}",
        "var_RH": "on", # 상대습도(RH) 필터 활성화
    }
    params.update(zip(LEVELS, VALUES))
    
    download_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{level}.pl"
    
    try:
        response = requests.get(download_url, params=params, timeout=30)
        output_file = f"{in_dir}/{sub_dir}/GFS_RH_{date}_{hours}_{level}_{time_offset}.grib2"
        
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return output_file
        else:
            print(f"Failed to download: {response.status_code}")
            return None
    except Exception as e:
        print(f"Download error: {e}")
        return None

def downsample_data(data):
    """(721, 1440) -> (181, 360) 다운샘플링"""
    new_lat, new_lon = 181, 360
    data_1deg = np.zeros((new_lat, new_lon))
    for i in range(new_lat - 1):
        for j in range(new_lon):
            data_1deg[i, j] = np.mean(data[i*4:i*4+4, j*4:j*4+4])
    for j in range(new_lon):
        data_1deg[-1, j] = np.mean(data[-1, j*4:j*4+4])
    return np.round(data_1deg, 2)

def get_rh_data(grbs, level_key):
    print(f"Extracting RH data for level: {level_key}")
    grbs.seek(0)
    p = PARAMS_FOR_LEVELS[level_key]
    
    rh_raw = None
    for grb in grbs:
        # GRIB 메시지 매칭 (이름, 레벨 타입, 레벨 값)
        if grb.name == p["RH_name"] and grb.typeOfLevel == p["typeOfLevel"] and grb.level == p["level"]:
            rh_raw = grb.values
            break

    if rh_raw is not None:
        if DOWNSAMPLE:
            rh_final = downsample_data(rh_raw)
            nx, ny, res = 360, 181, 1.0
        else:
            rh_final = np.round(rh_raw, 2)
            ny, nx = rh_final.shape
            res = 0.25

        return {
            "header": {
                "parameterName": "Relative Humidity",
                "parameterUnit": "%",
                "nx": nx, "ny": ny,
                "lo1": 0.0, "la1": 90.0,
                "lo2": 359.75 if res == 0.25 else 359.0,
                "la2": -90.0,
                "dx": res, "dy": res
            },
            "data": rh_final.flatten().tolist()
        }
    return None

def add_hours(dt_str, hours):
    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    dt = dt + timedelta(hours=hours)
    return dt.strftime('%Y%m%d%H%M')

# 메인 실행 루프
data_res = "0p25"
time_offsets = ["000", "001", "002", "003"]

while True:
    date, hour = get_utc_date_time()
    for offset in time_offsets:
        base_utc = f"{date}{hour}00"
        utc_str = add_hours(base_utc, int(offset))
        kor_str = add_hours(utc_str, 9)
        sub_dir = f"{kor_str[:4]}-{kor_str[4:6]}-{kor_str[6:8]}"
        
        os.makedirs(f"{in_dir}/{sub_dir}", exist_ok=True)
        os.makedirs(f"{out_dir}/{sub_dir}", exist_ok=True)

        grbs_file = download_file(date, hour, in_dir, sub_dir, level=data_res, time_offset=offset)
        if grbs_file:
            try:
                grbs = pygrib.open(grbs_file)
                for level_key in LEVELS:
                    result = get_rh_data(grbs, level_key)
                    if result:
                        fname = PARAMS_FOR_LEVELS[level_key]["fname_level"]
                        res_prefix = "025" if not DOWNSAMPLE else "100"
                        # 파일명에 rh 명시
                        target_path = f"{out_dir}/{sub_dir}/gfs_rh_{fname}_{res_prefix}_{utc_str}_{kor_str}.json"
                        with open(target_path, 'w') as f:
                            json.dump([result], f)
                        print(f"### Saved RH Data: {os.path.basename(target_path)}")
                grbs.close()
            except Exception as e:
                print(f"Error processing GRIB file: {e}")

    print("Process finished. Sleeping for 10 minutes...")
    time.sleep(600)