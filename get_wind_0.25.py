import pygrib
import numpy as np
import json
from datetime import datetime, timezone, timedelta
import requests
import os
import time
from config import get_config

# ==========================================
# 설정: 0.25도 데이터를 그대로 사용할지 여부
# True: 1.0도(360x181)로 변환 / False: 0.25도(1440x721) 유지
DOWNSAMPLE = False 
# ==========================================

config = get_config()
LEVELS = ["lev_10_m_above_ground","lev_850_mb", "lev_500_mb"]
PARAMS_FOR_LEVELS = {
  "lev_10_m_above_ground": {
    "U_name": "10 metre U wind component",
    "V_name": "10 metre V wind component",
    "typeOfLevel": "heightAboveGround",
    "level": 10,
    "fname_level": '10m'
  },
  "lev_500_mb": {
    "U_name": "U component of wind",
    "V_name": "V component of wind",
    "typeOfLevel": "isobaricInhPa",
    "level": 500,
    "fname_level": '500mb'
  },
  "lev_850_mb": {
    "U_name": "U component of wind",
    "V_name": "V component of wind",
    "typeOfLevel": "isobaricInhPa",
    "level": 850,
    "fname_level": '850mb'
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
    print(f"Downloading GFS wind data for {date} {hours}Z, level: {level}, time_offset: {time_offset}")
    params = {
        "dir": f"/gfs.{date}/{hours}/atmos/",
        "file": f"gfs.t{hours}z.pgrb2.{level}.f{time_offset}",
        "var_VGRD": "on",
        "var_UGRD": "on",
    }
    params.update(zip(LEVELS, VALUES))
    download_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{level}.pl"
    response = requests.get(download_url, params=params)
    output_File = f"{in_dir}/{sub_dir}/GFS_WIND_{date}_{hours}_{level}_{time_offset}.grib2"
    if response.status_code == 200:
        with open(output_File, 'wb') as f:
            f.write(response.content)
        return output_File
    return None

def downsample_to_1deg(ugrd_data, vgrd_data):
    # (721, 1440) -> (181, 360)
    new_lat, new_lon = 181, 360
    ugrd_1deg = np.zeros((new_lat, new_lon))
    vgrd_1deg = np.zeros((new_lat, new_lon))

    for i in range(new_lat - 1):
        for j in range(new_lon):
            ugrd_1deg[i, j] = np.mean(ugrd_data[i*4:i*4+4, j*4:j*4+4])
            vgrd_1deg[i, j] = np.mean(vgrd_data[i*4:i*4+4, j*4:j*4+4])
    
    # Handle South Pole
    for j in range(new_lon):
        ugrd_1deg[-1, j] = np.mean(ugrd_data[-1, j*4:j*4+4])
        vgrd_1deg[-1, j] = np.mean(vgrd_data[-1, j*4:j*4+4])

    return np.round(ugrd_1deg, 2), np.round(vgrd_1deg, 2)

def get_wind_data(grbs, LEVEL):
    grbs.seek(0)
    u_name = PARAMS_FOR_LEVELS[LEVEL]["U_name"]
    v_name = PARAMS_FOR_LEVELS[LEVEL]["V_name"]
    typeOfLevel = PARAMS_FOR_LEVELS[LEVEL]["typeOfLevel"]
    level_val = PARAMS_FOR_LEVELS[LEVEL]["level"]

    ugrd_raw, vgrd_raw = None, None

    for grb in grbs:
        if grb.name == u_name and grb.typeOfLevel == typeOfLevel and grb.level == level_val:
            ugrd_raw = grb.values
        if grb.name == v_name and grb.typeOfLevel == typeOfLevel and grb.level == level_val:
            vgrd_raw = grb.values

    if ugrd_raw is not None and vgrd_raw is not None:
        if DOWNSAMPLE:
            u_final, v_final = downsample_to_1deg(ugrd_raw, vgrd_raw)
            nx, ny, res = 360, 181, 1.0
        else:
            u_final, v_final = np.round(ugrd_raw, 2), np.round(vgrd_raw, 2)
            ny, nx = u_final.shape # 721, 1440
            res = 0.25

        results = []
        for name, data in [(u_name, u_final), (v_name, v_final)]:
            results.append({
                "header": {
                    "parameterName": name,
                    "numberPoints": nx * ny,
                    "nx": nx,
                    "ny": ny,
                    "lo1": 0.0,
                    "la1": 90.0,
                    "lo2": 359.75 if res == 0.25 else 359.0,
                    "la2": -90.0,
                    "dx": res,
                    "dy": res
                },
                "data": data.flatten().tolist()
            })
        return results
    return None

def add_hours(dt_str, hours):
    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    dt = dt + timedelta(hours=hours)
    return dt.strftime('%Y%m%d%H%M')

# 메인 루프
data_levels = ["0p25"]
time_offsets = {"0p25": ["000", "001", "002", "003", "004", "005", "006"]}

while True:
    date, hour = get_utc_date_time()
    job_list = []
    for level in data_levels:
        for time_offset in time_offsets[level]:
            job_list.append((date, hour, level, time_offset))

    while len(job_list) > 0:
        date, hour, level, time_offset = job_list.pop(0)
        base_utc = f"{date}{hour}00"
        utc_str = add_hours(base_utc, int(time_offset))
        kor_str = add_hours(utc_str, 9)
        sub_dir = f"{kor_str[:4]}-{kor_str[4:6]}-{kor_str[6:8]}"
        
        os.makedirs(f"{in_dir}/{sub_dir}", exist_ok=True)
        os.makedirs(f"{out_dir}/{sub_dir}", exist_ok=True)

        grbs_file = download_file(date, hour, in_dir, sub_dir, level=level, time_offset=time_offset)
        if grbs_file:
            grbs = pygrib.open(grbs_file)
            for LEVEL_KEY in LEVELS:
                results = get_wind_data(grbs, LEVEL_KEY)
                if results:
                    fname = PARAMS_FOR_LEVELS[LEVEL_KEY]["fname_level"]
                    res_prefix = "025" if not DOWNSAMPLE else "100"
                    target_path = f"{out_dir}/{sub_dir}/gfs_wind_{fname}_{res_prefix}_{utc_str}.json"
                    with open(target_path, 'w') as f:
                        json.dump(results, f)
                    print(f"### Saved: {target_path}")
            grbs.close()

    print("Sleeping for 10 minutes...")
    time.sleep(10)