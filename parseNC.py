import netCDF4 as nc
import numpy as np
from pyproj import Proj
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dataclasses import dataclass

@dataclass
class ProjectionAttributesLC:
  central_meridian: float     # 중앙 경도
  standard_parallel1: float   # 제1 표준 위도
  standard_parallel2: float   # 제2 표준 위도
  origin_latitude: float      # 원점 위도
  false_easting: float        # 가짜 동쪽
  false_northing: float       # 가짜 북쪽
  pixel_size: float           # 픽셀 크기 (미터 단위)
  upper_left_easting: float   # 좌상단 동쪽 좌표
  upper_left_northing: float  # 좌상단 북쪽 좌표

@dataclass
class ProjectionAttributesGE:
  sub_lon: float                  # 도를 라디안으로 변환
  perspective_point_height: float # 위성 높이 (km 단위로 변환)
  semi_major_axis: float          # 지구 적도 반지름 (미터)
  semi_minor_axis: float          # 지구 극 반지름 (미터)
  cfac: float                     # 열 방향 계수
  lfac: float                     # 행 방향 계수
  coff: float                     # 열 중심
  loff: float                     # 행 중심

_PLOT_BOUND = {
  'fd': [-180, 190, -80, 80],
  'ea': [70, 180, 0, 80]
}

def get_params_lc(file_path, var_name, grid_mapping):
  ds = nc.Dataset(file_path, 'r')

  attr_raw = ds.variables[var_name][:]
  dim_y, dim_x = attr_raw.shape
  print(f"{var_name} shape:", attr_raw.shape)

  projection = ds
  if grid_mapping != None:
    projection = ds.variables[grid_mapping]

  # 투영 정보 가져오기 (LCC)
  central_meridian = projection.getncattr("central_meridian")  # 126.0
  standard_parallel1 = projection.getncattr("standard_parallel1")  # 30.0
  standard_parallel2 = projection.getncattr("standard_parallel2")  # 60.0
  origin_latitude = projection.getncattr("origin_latitude")  # 38.0
  false_easting = projection.getncattr("false_easting")  # 0.0
  false_northing = projection.getncattr("false_northing")  # 0.0
  pixel_size = projection.getncattr("pixel_size")  # 2000.0 (미터 단위)
  upper_left_easting = projection.getncattr("upper_left_easting")  # -2999000.0
  upper_left_northing = projection.getncattr("upper_left_northing")  # 2599000.0
  projAttrs = ProjectionAttributesLC(
    central_meridian,
    standard_parallel1,
    standard_parallel2,
    origin_latitude,
    false_easting,
    false_northing,
    pixel_size,
    upper_left_easting,
    upper_left_northing
  )
  return attr_raw, dim_x, dim_y, projAttrs

def latlon_from_lc(projAttrs: ProjectionAttributesLC):
   # LCC 투영 정의
  proj = Proj(
      proj="lcc",
      lat_1=projAttrs.standard_parallel1,
      lat_2=projAttrs.standard_parallel2,
      lat_0=projAttrs.origin_latitude,
      lon_0=projAttrs.central_meridian,
      x_0=projAttrs.false_easting,
      y_0=projAttrs.false_northing,
      units="m",
      ellps="WGS84"
  )

  # 각 픽셀의 easting, northing 좌표 계산
  easting = np.linspace(projAttrs.upper_left_easting, projAttrs.upper_left_easting + projAttrs.pixel_size * (dim_x - 1), dim_x)
  northing = np.linspace(projAttrs.upper_left_northing, projAttrs.upper_left_northing - projAttrs.pixel_size * (dim_y - 1), dim_y)
  easting_grid, northing_grid = np.meshgrid(easting, northing)

  lon_grid, lat_grid = proj(easting_grid, northing_grid, inverse=True)
  return lon_grid, lat_grid

def parseLc(step, dim_x, dim_y, attr_raw, projAttrs):
  print("start parseLc: prams =", step, dim_x, dim_y)
  lon_grid, lat_grid = latlon_from_lc(projAttrs)
  result = []
  for y in range(0, dim_y, step):
    for x in range(0, dim_x, step):
      attr_value = attr_raw[y, x]
      result.append([
          float(lon_grid[y, x]),  # 경도
          float(lat_grid[y, x]),  # 위도
          float(attr_value)  # 픽셀 값 (ushort -> int로 변환)
      ])
  return result



def get_params_geos(file_path, var_name, grid_mapping):
  # NetCDF 파일 열기
  ds = nc.Dataset(file_path, 'r')

  # 이미지 픽셀 값과 차원 가져오기
  # ctt_raw = ds.variables['CTT'][:]  # CTT 데이터 (raw 값)
  attr_raw = ds.variables[var_name][:]  # CTT 데이터 (raw 값)
  dim_y, dim_x = attr_raw.shape
  print(f"{var_name} shape:", attr_raw.shape)

  # GEOS 투영 파라미터 (gk2a_imager_projection 변수에서 추출)
  projection = ds
  if grid_mapping != None:
    projection = ds.variables[grid_mapping]
  sub_lon_deg = projection.getncattr('longitude_of_projection_origin')  # 위성 경도 (도 단위)
  sub_lon = sub_lon_deg * math.pi / 180.0  # 도를 라디안으로 변환
  H = projection.perspective_point_height / 1000.0     # 위성 높이 (km 단위로 변환)
  a = projection.getncattr('semi_major_axis')          # 지구 적도 반지름 (미터)
  b = projection.getncattr('semi_minor_axis')          # 지구 극 반지름 (미터)
  cfac = projection.getncattr('column_scale_factor')   # 열 방향 계수
  lfac = projection.getncattr('line_scale_factor')     # 행 방향 계수
  coff = projection.getncattr('column_offset')         # 열 중심
  loff = projection.getncattr('line_offset')           # 행 중심
  print(f"sub_lon (deg): {sub_lon_deg}, sub_lon (rad): {sub_lon}, H: {H} km, a: {a} m, b: {b} m, cfac: {cfac}, lfac: {lfac}, coff: {coff}, loff: {loff}")
  projAttrs = ProjectionAttributesGE(
    sub_lon,
    H,
    a,
    b,
    cfac,
    lfac,
    coff,
    loff
  )
  return attr_raw, dim_x, dim_y, projAttrs

   
def latlon_from_geos(Line, Column, projAttrs, dim_x, dim_y):
  degtorad = 3.14159265358979 / 180.0
  x = degtorad * ((Column - projAttrs.coff) * 2**16 / projAttrs.cfac)  # 열 좌표
  y = degtorad * ((dim_y - Line - 1 - projAttrs.loff) * 2**16 / projAttrs.lfac)  # 행 좌표 (뒤집기 적용)
  # 모든 좌표에 대해 x, y 출력 (디버깅용)
  # print(f"Line={Line}, Column={Column}, x={x}, y={y}")
  # Sd 계산에서 기존 스크립트와 동일한 스케일링 적용
  # Sd = np.sqrt((H * np.cos(x) * np.cos(y))**2 - (np.cos(y)**2 + 1.006739501 * np.sin(y)**2) * (a**2 / (1000 * 1000)))
  Sd = np.sqrt((42164.0 * np.cos(x) * np.cos(y))**2 - (np.cos(y)**2 + 1.006739501 * np.sin(y)**2) * 1737122264)
  if Sd < 0:  # 무효값 방지 및 디버깅
    print(f"Sd negative: {Sd} at (x={x}, y={y}, Line={Line}, Column={Column})")
    return None, None
  Sn = (projAttrs.perspective_point_height * np.cos(x) * np.cos(y) - Sd) / (np.cos(y)**2 + 1.006739501 * np.sin(y)**2)
  S1 = projAttrs.perspective_point_height - (Sn * np.cos(x) * np.cos(y))
  S2 = Sn * (np.sin(x) * np.cos(y))
  S3 = -Sn * np.sin(y)
  Sxy = np.sqrt((S1 * S1) + (S2 * S2))
  nlon = (np.arctan(S2 / S1) + projAttrs.sub_lon) / degtorad
  nlat = np.arctan((1.006739501 * S3) / Sxy) / degtorad
  return nlon, nlat

def parseGeos(step, dim_x, dim_y, attr_raw, projAttrs):
  result = []
  for y in range(0, dim_y, step):
    for x in range(0, dim_x, step):
      ctt_value = attr_raw[y, x]
      # _FillValue(65535) 제외 및 유효 범위 확인
      if ctt_value != 65535 and 0 <= ctt_value <= 35000:  # CTT의 _FillValue와 유효 범위
        lon, lat = latlon_from_geos(y, x, projAttrs, dim_x, dim_y)
        if lon is not None and lat is not None:
          if -80 <= lat <= 80 and -180 <= lon <= 190:  # 남/북 범위 확장
            # CTT 값 스케일링 적용 (K 단위)
            # ctt_adjusted = ctt_value * 0.01  # scale_factor = 0.01, add_offset = 0.0
            ctt_adjusted = ctt_value # scale_factor = 0.01, add_offset = 0.0
            result.append([
                float(lon),  # 경도
                float(lat),  # 위도
                float(ctt_adjusted)  # CTT 값 (K 단위)
            ])
  return result

def save_to_file(out_file, json_data):
  with open(out_file, "w") as f:
    json.dump(json_data, f)
  print(f"데이터가 {out_file}에 저장되었습니다. 총 {len(json_data)}개 포인트")
  return True

def desctruct_att_lat_lon(attr_with_lat_lon):
  lons = [point[0] for point in attr_with_lat_lon]
  lats = [point[1] for point in attr_with_lat_lon]
  attr_values = [point[2] for point in attr_with_lat_lon]
  return lons, lats, attr_values

def print_result(lons, lats, attr_values):
  print("Longitude 범위:", min(lons) if lons else "No valid points", "to", max(lons) if lons else "No valid points")
  print("Latitude 범위:", min(lats) if lats else "No valid points", "to", max(lats) if lats else "No valid points")
  print(f"{attr_to_get} 범위:", min(attr_values) if attr_values else "No valid points", "to", max(attr_values) if attr_values else "No valid points")

def show_plot(lons, lats, attr_values, nc_coverage, plot_title):
  plt.figure(figsize=(10, 8))
  ax = plt.axes(projection=ccrs.PlateCarree())
  ax.set_extent(_PLOT_BOUND[nc_coverage], crs=ccrs.PlateCarree())  # 남/북 범위 확장
  ax.add_feature(cfeature.COASTLINE)
  ax.add_feature(cfeature.BORDERS, linestyle=':')
  ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
  ax.add_feature(cfeature.LAND, facecolor='lightgreen')

  scatter = ax.scatter(lons, lats, c=attr_values, cmap='viridis', s=1, transform=ccrs.PlateCarree())
  plt.colorbar(scatter)
  plt.title(plot_title)
  plt.show()



if __name__ == '__main__' :
  step = 10
  basename = Path(nc_file).stem
  nc_coverage = basename.split('_')[1]
  out_dir = './jsonfiles'
  out_file = f"{out_dir}/{basename}_step{step}.json"
  parseResult = []
  GRID_MAPPING = {
    "ctps": 'gk2a_imager_projection', # if data is ctps, grid_mapping = gk2a_imager_projection
    "ir105": None
  }

  # test ctps fd
  # nc_file = './working_script_samples/ctps_fd_ge_202502170000.nc'
  # attr_to_get = 'CTT'
  # attr_raw, dim_x, dim_y, projAttrs = get_params_geos(nc_file, attr_to_get, GRID_MAPPING.ctps)
  # parseResult = parseGeos(step, dim_x, dim_y, attr_raw, projAttrs)

  # test ctps lc
  # nc_file = './working_script_samples/ctps_ea_lc_202502170000.nc'
  # attr_to_get = 'CTT'
  # attr_raw, dim_x, dim_y, projAttrs = get_params_lc(nc_file, attr_to_get, GRID_MAPPING.get('ctps', None))
  # parseResult = parseLc(step, dim_x, dim_y, attr_raw, projAttrs)

  # test ri105 ea
  nc_file = './working_script_samples/ir105_ea_lc_202502170000.nc'
  attr_to_get = 'image_pixel_values' # for ri105
  attr_raw, dim_x, dim_y, projAttrs = get_params_lc(nc_file, attr_to_get, GRID_MAPPING.get('ir105', None))
  print(projAttrs.central_meridian, projAttrs.upper_left_easting)
  parseResult = parseLc(step, dim_x, dim_y, attr_raw, projAttrs)

  print("parse result Done:", len(parseResult))

  # common code
  lons, lats, attr_values = desctruct_att_lat_lon(parseResult);

  print_result(lons, lats, attr_values)
  save_to_file(out_file, parseResult)
  show_plot(lons, lats, attr_values, nc_coverage, f"visualization: {out_file}")

