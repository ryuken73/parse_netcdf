import pygrib
import numpy as np
import pandas as pd
import matplotlib
import json
from datetime import datetime, timezone, timedelta
import requests
import os
import time
from config import get_config

config = get_config()
print(f'Running in {config.ENV} mode')
print(f'OUT_PATH_WIND = {config.OUT_PATH_WIND}')
print(f'WATCH_PATH_WIND = {config.WATCH_PATH_WIND}')

BASE_GFS_URL = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl"
LEVELS = ["lev_10_m_above_ground"]
VALUES = ["on"] * len(LEVELS)

in_dir = config.WATCH_PATH_WIND
out_dir = config.OUT_PATH_WIND

def get_utc_date_time():
  # 현재 UTC 시간 가져오기
  now = datetime.now(timezone.utc)
  # 날짜를 %Y%m%d 형식으로 변환
  date_str = now.strftime("%Y%m%d")
  # 가능한 시간 목록
  valid_hours = [0, 6, 12, 18]
  # 현재 시간(시간 단위)
  current_hour = now.hour
  # 가장 가까운 과거 시간 찾기
  past_hour = max([h for h in valid_hours if h <= current_hour], default=18)
  # 시간을 두 자리 문자열로 변환 (예: "06")
  time_str = f"{past_hour:02d}"
  return date_str, time_str

def download_file(date, hours, in_dir, sub_dir):
  params = {
    "dir": f"/gfs.{date}/{hours}/atmos/",
    "file": f"gfs.t{hours}z.pgrb2.1p00.f000",
    "var_VGRD": "on",
    "var_UGRD": "on",
  }
  params.update(zip(LEVELS, VALUES))
  print(f"Downloading GFS data for date hours {date} {hours}Z...")
  response = requests.get(BASE_GFS_URL, params=params)
  output_File = f"{in_dir}/{sub_dir}/GFS_WIND_{date}.grib2"
  if response.status_code == 200:
    with open(output_File, 'wb') as f:
      f.write(response.content)
    return output_File
  else:
    print(f"Failed to download file: {response.status_code}")
    return None

def show_grib_file(grbs_file):
  grbs = pygrib.open(grbs_file)
  # print all variables
  print('Available variables and levels')
  print(grbs.tell())
  for grb in grbs:
    print(f"Variable: {grb.name}, Level: {grb.level} {grb.typeOfLevel}, Time: {grb.validDate}")
  # should locate 0 to iterate again
  grbs.seek(0)
  return grbs

def get_wind_data(grbs):
  # ugrd와 vgrd (10m wind) 추출
  ugrd_data = None;
  vgrd_data = None;
  lats, lons = None, None;

  for grb in grbs:
    if grb.name == "10 metre U wind component" and grb.typeOfLevel == "heightAboveGround" and grb.level == 10 :
      ugrd_data = np.round(grb.values, 2)
      lats, lons = grb.latlons()
    if grb.name == "10 metre V wind component" and grb.typeOfLevel == "heightAboveGround" and grb.level == 10 :
      vgrd_data = np.round(grb.values, 2)

  def flatten_data(grd_data):
    return grd_data.T.flatten().tolist()

  # check data
  print(ugrd_data, vgrd_data)
  if ugrd_data is not None and vgrd_data is not None:
    print(f"UGRD shape: {ugrd_data.shape}, VGRD shape: {vgrd_data.shape}")
    print(f"UGRD Min: {ugrd_data.min()}, VGRD Min: {vgrd_data.min()}")
    print(f"UGRD Max: {ugrd_data.max()}, VGRD Max: {vgrd_data.max()}")
    print(f"Lat and Lon range: {lats.min():.2f} to {lats.max():.2f}, {lons.min():.2f} to {lons.max():.2f}")
    # calculate wind speed
    wind_speed = np.sqrt(ugrd_data**2 + vgrd_data**2)
    print(f"Wind Speed Range: {wind_speed.min():.2f}, Max: {wind_speed.max():.2f}")

    results = [
      {
        "header": {
          "refTime": "2025-08-30 00:00:00",
          "parameterName": 'U wind component',
          "numberPoints": 65160,
          "surface1TypeName": "Specified height level above ground",
          "surface1Value": 10.0,
          "nx": 360,
          "ny": 181,
          "lo1": 0.0,
          "la1": 90.0,
          "lo2": 359.0,
          "la2": -90.0,
          "dx": 1.0,
          "dy": 1.0
        },
        "data": ugrd_data.flatten().tolist()
      },
      {
        "header": {
          "refTime": "2025-08-30 00:00:00",
          "parameterName": 'V wind component',
          "numberPoints": 65160,
          "surface1TypeName": "Specified height level above ground",
          "surface1Value": 10.0,
          "nx": 360,
          "ny": 181,
          "lo1": 0.0,
          "la1": 90.0,
          "lo2": 359.0,
          "la2": -90.0,
          "dx": 1.0,
          "dy": 1.0
        },
        "data": vgrd_data.flatten().tolist()
      }
    ]
  return results;

def add_hours(dt_str, hours):
  # 문자열을 datetime 객체로 변환
  dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S')
  # 9시간 추가
  dt = dt + timedelta(hours=hours)
  # 다시 문자열 형식으로 변환
  return dt.strftime('%Y%m%d%H%M%S')

# open GRIB2
# grbs_file = "gfs.t00z.pgrb2.0p25.f000"
# grbs_file = "gfs.t00z.pgrb2.1p00.f000"
date_time_queue = [
  ["20250902", "00"],
  ["20250902", "06"],
  ["20250902", "12"],
  ["20250902", "18"],
  ["20250903", "00"],
  ["20250903", "06"],
]

while True:
  date, hour = get_utc_date_time()
  # save date, hour to date_time_queue if not exists
  if (date, hour) not in date_time_queue:
    date_time_queue.append((date, hour))
  # for each date, hour in date_time_queue, download and makes json file
  date_time_done = []
  for dt in date_time_queue:
    print(f"### Start Processing {dt[0]} {dt[1]} {datetime.now()}")
    utc_string = f"{dt[0]}{dt[1]}0000"
    kor_string = add_hours(utc_string, 9)
    sub_dir = f"{kor_string[:4]}-{kor_string[4:6]}-{kor_string[6:8]}"
    os.makedirs(f"{in_dir}/{sub_dir}", exist_ok=True)
    os.makedirs(f"{out_dir}/{sub_dir}", exist_ok=True)
    target_file = f"{out_dir}/{sub_dir}/gfs_wind_10m_{utc_string}_{kor_string}.json"

    # if target_file already exists, continue
    if os.path.exists(target_file):
      date_time_done.append(dt)
      print(f"%%% {target_file} already exists, skipping...")
      continue
    grbs_file = download_file(dt[0], dt[1], in_dir, sub_dir)
    if grbs_file is None:
      print(f"%%% Will retry downloading later for {dt[0]} {dt[1]}")
      continue
    grbs = show_grib_file(grbs_file)
    results = get_wind_data(grbs)

    with open(target_file, 'w') as f:
      json.dump(results, f)
    print(f"%%% Data saved to {target_file}")
    date_time_done.append(dt)
    # remove processed date, hour from date_time_queue
    # date_time_queue.remove(dt)
  # sleep 10m
  for dt in date_time_done:
    print(f"%%% Finished Processing {dt[0]} {dt[1]} {datetime.now()}")
    if dt in date_time_queue:
      print(f'remove dt {dt}')
      date_time_queue.remove(dt)
  print(f"Current date/time queue: {date_time_queue}")
  print("Sleeping for 10 minutes...")
  time.sleep(600)