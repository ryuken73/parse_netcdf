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

def download_file(date, hours, in_dir, sub_dir, level=1, time_offset="000"):
  params = {
    "dir": f"/gfs.{date}/{hours}/atmos/",
    "file": f"gfs.t{hours}z.pgrb2.{level}.f{time_offset}",
    "var_VGRD": "on",
    "var_UGRD": "on",
  }
  params.update(zip(LEVELS, VALUES))
  print(f"Downloading GFS data for date hours {date} {hours}Z {level} {time_offset}...")
  download_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{level}.pl"
  print(download_url, params)
  response = requests.get(download_url, params=params)
  output_File = f"{in_dir}/{sub_dir}/GFS_WIND_{date}_{hours}_{level}_{time_offset}.grib2"
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

def downsample_to_1deg(ugrd_data, vgrd_data, lats, lons):
  """
  Downsample 0.25° GFS data (721x1440) to 1° resolution (181x360) for UGRD, VGRD, and wind speed.
  """
  # Original dimensions: 721 (lat) x 1440 (lon)
  orig_lat, orig_lon = ugrd_data.shape  # (721, 1440)
  new_lat, new_lon = 181, 360  # Target 1° resolution

  # Initialize output arrays
  ugrd_1deg = np.zeros((new_lat, new_lon))
  vgrd_1deg = np.zeros((new_lat, new_lon))
  wind_speed_1deg = np.zeros((new_lat, new_lon))

  # Downsample by averaging 4x4 blocks (0.25° to 1°)
  for i in range(new_lat - 1):  # Process 180 lat points (excluding last point)
    for j in range(new_lon):
      # Calculate indices for 4x4 block
      lat_start = i * 4
      lon_start = j * 4
      # Average 4x4 block for UGRD and VGRD
      ugrd_1deg[i, j] = np.mean(ugrd_data[lat_start:lat_start+4, lon_start:lon_start+4])
      vgrd_1deg[i, j] = np.mean(vgrd_data[lat_start:lat_start+4, lon_start:lon_start+4])

  # Handle the last latitude point (90°S, 721st point) as a single value
  for j in range(new_lon):
    lon_start = j * 4
    # Use the single value at the last latitude point
    ugrd_1deg[new_lat-1, j] = np.mean(ugrd_data[-1, lon_start:lon_start+4])
    vgrd_1deg[new_lat-1, j] = np.mean(vgrd_data[-1, lon_start:lon_start+4])

  # Calculate wind speed from downsampled UGRD and VGRD
  wind_speed_1deg = np.sqrt(ugrd_1deg**2 + vgrd_1deg**2)

  # Round to 2 decimal places
  ugrd_1deg = np.round(ugrd_1deg, 2)
  vgrd_1deg = np.round(vgrd_1deg, 2)
  wind_speed_1deg = np.round(wind_speed_1deg, 2)

  # Generate new lat/lon arrays for 1° grid
  lats_1deg = np.linspace(90, -90, new_lat)  # 90°N to 90°S
  lons_1deg = np.linspace(0, 359, new_lon)   # 0°E to 359°E

  return ugrd_1deg, vgrd_1deg, wind_speed_1deg, lats_1deg, lons_1deg

def flatten_data(grd_data):
  """Flatten 2D grid data to 1D list in transposed order for JSON compatibility."""
  return grd_data.T.flatten().tolist()

def get_wind_data(grbs, LEVEL):
  # locate first position of gribs
  grbs.seek(0)
  # ugrd와 vgrd (10m wind) 추출
  print(f"get_wind_data for {LEVEL}")
  ugrd_data = None;
  vgrd_data = None;
  lats, lons = None, None;
  u_name = PARAMS_FOR_LEVELS[LEVEL]["U_name"]
  v_name = PARAMS_FOR_LEVELS[LEVEL]["V_name"]
  typeOfLevel = PARAMS_FOR_LEVELS[LEVEL]["typeOfLevel"]
  level = PARAMS_FOR_LEVELS[LEVEL]["level"]

  for grb in grbs:
    print(grb.name, grb.typeOfLevel, grb.level)
    # if grb.name == "10 metre U wind component" and grb.typeOfLevel == "heightAboveGround" and grb.level == 10 :
    if grb.name == u_name and grb.typeOfLevel == typeOfLevel and grb.level == level :
      ugrd_data = np.round(grb.values, 2)
      lats, lons = grb.latlons()
    # if grb.name == "10 metre V wind component" and grb.typeOfLevel == "heightAboveGround" and grb.level == 10 :
    if grb.name == v_name and grb.typeOfLevel == typeOfLevel and grb.level == level :
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
    ugrd_1deg, vgrd_1deg, wind_speed_1deg, lats_1deg, lons_1deg = downsample_to_1deg(ugrd_data, vgrd_data, lats, lons)

    # Print diagnostics
    print(f"Downsampled UGRD shape: {ugrd_1deg.shape}, VGRD shape: {vgrd_1deg.shape}")
    print(f"Downsampled UGRD Min: {ugrd_1deg.min()}, VGRD Min: {vgrd_1deg.min()}")
    print(f"Downsampled UGRD Max: {ugrd_1deg.max()}, VGRD Max: {vgrd_1deg.max()}")
    print(f"Downsampled Wind Speed Range: {wind_speed_1deg.min():.2f}, Max: {wind_speed_1deg.max():.2f}")
    print(f"Downsampled Lat range: {lats_1deg.min():.2f} to {lats_1deg.max():.2f}")
    print(f"Downsampled Lon range: {lons_1deg.min():.2f} to {lons_1deg.max():.2f}")

    # calculate wind speed
    # wind_speed = np.sqrt(ugrd_data**2 + vgrd_data**2)
    # print(f"Wind Speed Range: {wind_speed.min():.2f}, Max: {wind_speed.max():.2f}")

    results = [
      {
        "header": {
          "refTime": "2025-08-30 00:00:00",
          "parameterName": u_name,
          "numberPoints": 65160,
          "surface1TypeName": "Specified height level above ground",
          "surface1Value": level,
          "nx": 360,
          "ny": 181,
          "lo1": 0.0,
          "la1": 90.0,
          "lo2": 359.0,
          "la2": -90.0,
          "dx": 1.0,
          "dy": 1.0
        },
        "data": ugrd_1deg.flatten().tolist()
      },
      {
        "header": {
          "refTime": "2025-08-30 00:00:00",
          "parameterName": v_name,
          "numberPoints": 65160,
          "surface1TypeName": "Specified height level above ground",
          "surface1Value": level,
          "nx": 360,
          "ny": 181,
          "lo1": 0.0,
          "la1": 90.0,
          "lo2": 359.0,
          "la2": -90.0,
          "dx": 1.0,
          "dy": 1.0
        },
        "data": vgrd_1deg.flatten().tolist()
      }
    ]
  return results;

def add_hours(dt_str, hours):
  # 문자열을 datetime 객체로 변환
  dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
  # 9시간 추가
  dt = dt + timedelta(hours=hours)
  # 다시 문자열 형식으로 변환
  return dt.strftime('%Y%m%d%H%M')

# open GRIB2
# grbs_file = "gfs.t00z.pgrb2.0p25.f000"
# grbs_file = "gfs.t00z.pgrb2.1p00.f000"
# date_time_queue = [
#   ["20250902", "00", "1", "000"],
#   ["20250902", "00", "1", "003"],
#   ["20250902", "00", "1", "006"],
#   ["20250902", "00", "0.25", "000"],
#   ["20250902", "00", "0.25", "001"],
#   ["20250902", "00", "0.25", "002"],
#   ["20250902", "00", "0.25", "003"],
#   ["20250902", "00", "0.25", "004"],
#   ["20250902", "00", "0.25", "005"],
#   ["20250902", "00", "0.25", "006"],
# ]

# data_levels = ["0p25", "1p00"];
data_levels = ["0p25"]

time_offsets = {
  "1p00": ["000", "003", "006"],
  "0p25": ["000", "001", "002", "003", "004", "005", "006"]
}

# use prev_date_hours to get missed hours
prev_date_hours = [
  # ("20250902", "00"),
  # ("20250902", "06"),
  # ("20250902", "12"),
  # ("20250902", "18"),
  # ("20250903", "00"),
  # ("20250903", "06"),
  # ("20250903", "12"),
  # ("20250903", "18"),
  # ("20250904", "00"),
  # ("20250904", "06"),
  # ("20250904", "12"),
  # ("20250904", "18"),
  # ("20250905", "00"),
]

def make_job_list(date, hour):
  # merge with prev_date_hours
  date_hours = prev_date_hours + [(date, hour)]
  print(prev_date_hours,date_hours)
  job_list = []
  for dt in date_hours:
    for level in data_levels:
      for time_offset in time_offsets[level]:
        job_list.append((dt[0], dt[1], level, time_offset))
  return job_list

while True:
  date, hour = get_utc_date_time()
  # save date, hour to date_time_queue if not exists
  # date = "20250902"
  # hour = "00"
  job_list = make_job_list(date, hour)
  while len(job_list) > 0:
    job = job_list.pop(0)
    date, hour, level, time_offset = job
    # process the job
    base_utc_string = f"{date}{hour}00"
    utc_string = add_hours(base_utc_string, int(time_offset))
    kor_string = add_hours(utc_string, 9)
    sub_dir = f"{kor_string[:4]}-{kor_string[4:6]}-{kor_string[6:8]}"
    os.makedirs(f"{in_dir}/{sub_dir}", exist_ok=True)
    os.makedirs(f"{out_dir}/{sub_dir}", exist_ok=True)
    # if 10m json exists, should not start download grib file
    target_file = f"{out_dir}/{sub_dir}/gfs_wind_10m_{utc_string}_{kor_string}.json"
    print(f"### Start Processing {target_file} level {level} {datetime.now()}")

    # if target_file already exists, continue
    if os.path.exists(target_file):
      print(f"%%% {target_file} already exists, skipping...")
      continue
    grbs_file = download_file(date, hour, in_dir, sub_dir, level=level, time_offset=time_offset)
    if grbs_file is None:
      print(f"%%% Will retry downloading later for {date} {hour} {level} {time_offset}")
      continue
    grbs = show_grib_file(grbs_file)
    for LEVEL in LEVELS:
      results = get_wind_data(grbs, LEVEL)
      fname_level = PARAMS_FOR_LEVELS[LEVEL]["fname_level"]
      target_file_level = f"{out_dir}/{sub_dir}/gfs_wind_{fname_level}_{utc_string}_{kor_string}.json"
      with open(target_file_level, 'w') as f:
        json.dump(results, f)
      print(f"%%% Data saved to {target_file_level}")

  print("Sleeping for 10 minutes...")
  time.sleep(600)