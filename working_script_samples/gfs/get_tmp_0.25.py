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
# True: 1.0도(360x181)로 변환 / False: 0.25도(1440x721) 유지
DOWNSAMPLE = False 
# ==========================================

config = get_config()

# 추출할 기상 레벨 정의 (온도 전용)
# LEVELS = ["lev_2_m_above_ground","lev_10_m_above_ground", "lev_850_mb", "lev_500_mb"] ## 10m는 온도 데이터에는 없음
LEVELS = ["lev_2_m_above_ground", "lev_850_mb", "lev_500_mb"]
PARAMS_FOR_LEVELS = {
    "lev_2_m_above_ground": {
        "TMP_name": "2 metre temperature",
        "typeOfLevel": "heightAboveGround",
        "level": 2, # GFS 지상 기온은 2m 레벨임
        "fname_level": "2m"
    },
    "lev_500_mb": {
        "TMP_name": "Temperature",
        "typeOfLevel": "isobaricInhPa",
        "level": 500,
        "fname_level": "500mb"
    },
    "lev_850_mb": {
        "TMP_name": "Temperature",
        "typeOfLevel": "isobaricInhPa",
        "level": 850,
        "fname_level": "850mb"
    },
}

# 필터 옵션 설정을 위해 LEVELS 개수만큼 "on" 생성
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
    print(f"Downloading GFS TMP data for {date} {hours}Z, F{time_offset}")
    params = {
        "dir": f"/gfs.{date}/{hours}/atmos/",
        "file": f"gfs.t{hours}z.pgrb2.{level}.f{time_offset}",
        "var_TMP": "on", # 온도 데이터만 필터링
    }
    # 각 고도 레벨 필터 추가
    params.update(zip(LEVELS, VALUES))
    
    download_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{level}.pl"
    response = requests.get(download_url, params=params)
    
    output_file = f"{in_dir}/{sub_dir}/GFS_TMP_{date}_{hours}_{level}_{time_offset}.grib2"
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        return output_file
    else:
        print(f"Failed to download: {response.status_code}")
        return None

def downsample_tmp(tmp_data):
    """(721, 1440) -> (181, 360) 다운샘플링"""
    new_lat, new_lon = 181, 360
    tmp_1deg = np.zeros((new_lat, new_lon))
    for i in range(new_lat - 1):
        for j in range(new_lon):
            tmp_1deg[i, j] = np.mean(tmp_data[i*4:i*4+4, j*4:j*4+4])
    # 남극점 처리
    for j in range(new_lon):
        tmp_1deg[-1, j] = np.mean(tmp_data[-1, j*4:j*4+4])
    return np.round(tmp_1deg, 2)

def get_tmp_data(grbs, level_key):
    print(f"Extracting TMP data for level: {level_key}")
    grbs.seek(0)
    p = PARAMS_FOR_LEVELS[level_key]
    
    tmp_raw = None
    for grb in grbs:
        print(f"Checking GRIB message: {grb.name}, level: {grb.level}, typeOfLevel: {grb.typeOfLevel}")
        if grb.name == p["TMP_name"] and grb.typeOfLevel == p["typeOfLevel"] and grb.level == p["level"]:
            tmp_raw = grb.values
            break

    if tmp_raw is not None:
        if DOWNSAMPLE:
            tmp_final = downsample_tmp(tmp_raw)
            nx, ny, res = 360, 181, 1.0
        else:
            tmp_final = np.round(tmp_raw, 2)
            ny, nx = tmp_final.shape
            res = 0.25

        return {
            "header": {
                "parameterName": p["TMP_name"],
                "nx": nx, "ny": ny,
                "lo1": 0.0, "la1": 90.0,
                "lo2": 359.75 if res == 0.25 else 359.0,
                "la2": -90.0,
                "dx": res, "dy": res
            },
            "data": tmp_final.flatten().tolist()
        }
    return None

def add_hours(dt_str, hours):
    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    dt = dt + timedelta(hours=hours)
    return dt.strftime('%Y%m%d%H%M')

# 메인 실행 루프
data_res = "0p25"
time_offsets = [
    "000", "001", "002", "003", "004", "005", "006", "007", "008", "009", 
    "010", "011", "012", "013", "014", "015", "016", "017", "018", "019", 
    "020", "021", "022", "023", "024", "025", "026", "027", "028", "029", 
    "030", "031", "032", "033", "034", "035", "036", "037", "038", "039", 
    "040", "041", "042", "043", "044", "045", "046", "047", "048", "049", 
    "050", "051", "052", "053", "054", "055", "056", "057", "058", "059", 
    "060", "061", "062", "063", "064", "065", "066", "067", "068", "069", 
    "070", "071", "072", "073", "074", "075", "076", "077", "078", "079",
    "080", "081", "082", "083", "084", "085", "086", "087", "088", "089",
    "090", "091", "092", "093", "094", "095", "096", "097", "098", "099",
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "110", "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "120"
]

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
            grbs = pygrib.open(grbs_file)
            for level_key in LEVELS:
                result = get_tmp_data(grbs, level_key)
                if result:
                    fname = PARAMS_FOR_LEVELS[level_key]["fname_level"]
                    res_prefix = "025" if not DOWNSAMPLE else "100"
                    target_path = f"{out_dir}/{sub_dir}/gfs_tmp_{fname}_{res_prefix}_{utc_str}_{kor_str}.json"
                    with open(target_path, 'w') as f:
                        json.dump([result], f) # 호환성을 위해 리스트로 감쌈
                    print(f"### Saved TMP Data: {os.path.basename(target_path)}")
            grbs.close()

    print("Process finished. Sleeping for 10 minutes...")
    time.sleep(600)