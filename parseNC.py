import netCDF4 as nc
import numpy as np
from pyproj import Proj
import os
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer, CRS
from PIL import Image
from dataclasses import dataclass
import gzip
import shutil

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

IMAGE_SIZE = {
  'ea': {
    'step4': (800, 700),
    'step10': (800, 700),
  },
  'fd': {
    'step3': (2048, 2048),
    'step10': (1200, 1200),
  }
}

# IMAGE_SIZE = {
#   # 'ea': (600, 520),
#   'ea': (800, 780),
#   'fd': (800, 780),
#   # 'ea': (1200, 1040),
# }

IMAGE_BOUNDS = {
  'ea': [76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447],
  'fd': [60, -80, 180, 80]
}

# Web Mercator 투영 정의
web_mercator_crs = CRS.from_epsg("3857")  # EPSG:3857 (Web Mercator)
wgs84_crs = CRS.from_epsg("4326")  # EPSG:4326 (WGS84)
transformer_to_mercator = Transformer.from_crs(wgs84_crs, web_mercator_crs, always_xy=True)

# 적절한 보간을 위해 포인트가 데이터 영역내의 결측데이터인지 여부를 판단
def is_point_off_range(grid_values, i, j, height, width):
    left_value = grid_values[i][j-1 if j > 1 else 0]
    right_value = grid_values[i][j+1 if j < width-1 else width-1]
    up_value = grid_values[i-1 if i > 1 else 0][j]
    up_left = grid_values[i-1 if i > 1 else 0][j-1 if j > 1 else 0]
    up_right = grid_values[i-1 if i > 1 else 0][j+1 if j < width-1 else width-1]
    down_value = grid_values[i+1 if i < height-1 else height-1][j]
    down_left = grid_values[i+1 if i < height-1 else height-1][j-1 if j > 1 else 0]
    down_right = grid_values[i+1 if i < height-1 else height-1][j+1 if j < width-1 else width-1]
    return left_value == -9999 and right_value == -9999 and up_value == -9999 and down_value == -9999 and up_left == -9999 and up_right == -9999 and down_left == -9999 and down_right == -9999

# 색상 매핑 (검정-흰색 보간)
def get_color_ir105_mono(value):
    if value == -9999:
        return [0, 0, 0, 0]  # 투명
        # print(f'fill red {value}')
        return [255, 0, 0, 255]  # red
    t_min, t_max = -100, 30
    ratio = min(max((value - t_min) / (t_max - t_min), 0), 1)
    r = int(255 * (1 - ratio))
    g = int(255 * (1 - ratio))
    b = int(255 * (1 - ratio))
    return [r, g, b, 255]  # RGBA

def get_color_ir105_color(temp):
    # Define key color points and their RGB values based on the gradient
    colors = {
        20: [0, 0, 0],       # Black
        -20: [255, 255, 255], # White
        -21: [135, 206, 235], # Sky blue (sharp transition from white at -20)
        -30: [0, 0, 255],    # Blue
        -40: [0, 255, 0],    # Green
        -45: [144, 238, 144], # Light green
        -50: [255, 255, 0],   # Yellow
        -60: [255, 0, 0],     # Red
        -70: [0, 0, 0],       # Black
        -80: [255, 255, 255], # White (sharp transition from black at -70)
        -81: [128, 128, 128], # Gray (sharp transition from white at -80)
        -90: [128, 0, 128]    # Purple
    }
  
    if temp > 20 or temp < -90 :
        return [0, 0, 0, 0]    
    # Clamp temperature to valid range (20 to -90)

    temp = max(-90, min(20, temp))
    
    # Find the two closest key points for interpolation (considering sharp transitions)
    keys = sorted(colors.keys(), reverse=True)  # Sort in descending order (20 to -90)
    for i in range(len(keys) - 1):
        if temp <= keys[i] and temp > keys[i + 1]:
            start_temp, end_temp = keys[i], keys[i + 1]
            start_color, end_color = colors[start_temp], colors[end_temp]
            break
    else:
        if temp <= -81:
            start_temp, end_temp = -81, -90
            start_color, end_color = colors[-81], colors[-90]
        elif temp >= 20:
            return [0, 0, 0, 255]  # Black for temp >= 20
        else:
            start_temp, end_temp = keys[0], keys[1]
            start_color, end_color = colors[start_temp], colors[end_temp]
    
    # Linear interpolation based on temperature position
    if start_temp == end_temp or (start_temp in [-20, -80] and temp == start_temp + 1):
        return end_color + [255]  # Sharp transition at -20 and -80
    else:
        ratio = (start_temp - temp) / (start_temp - end_temp)  # Adjust for descending order
        r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
        return [r, g, b, 255]

get_color_func = {
   'mono': get_color_ir105_mono,
   'color': get_color_ir105_color,
}

def save_to_image_ir105(data, output_path, nc_coverage, step, mode='mono'):
    """
    주어진 [[lon, lat, value], ...] 데이터를 Web Mercator 투영을 반영하여 이미지로 변환.
    """
    image_size = IMAGE_SIZE[nc_coverage][f'step{step}']
    bounds = IMAGE_BOUNDS[nc_coverage]
    print(image_size)
    # 경계 설정 (WGS84 좌표)
    lon_min, lat_min, lon_max, lat_max = bounds
    
    # 이미지 크기
    width, height = image_size  # 1200x1040 (원본 비율에 가까운 해상도)
    
    # WGS84 경계 좌표를 Web Mercator로 변환
    x_min, y_max = transformer_to_mercator.transform(lon_min, lat_max)  # 좌상단
    x_max, y_min = transformer_to_mercator.transform(lon_max, lat_min)  # 우하단
    # print(f"x_min, x_max", x_min, x_max)
    # print(f"y_min, y_max", y_min, y_max)
    
    # Web Mercator 좌표 간격 계산
    x_step = (x_max - x_min) / (width - 1)
    y_step = (y_max - y_min) / (height - 1)
    
    # 2D 배열 초기화 (값 저장용)
    grid_values = np.full((height, width), -9999, dtype=np.float32)
    
    # 데이터를 Web Mercator 좌표로 변환 후 매핑
    for lon, lat, value in data:
        # WGS84 -> Web Mercator 변환
        x, y = transformer_to_mercator.transform(lon, lat)
        # Web Mercator 좌표를 픽셀 인덱스로 변환
        col = int((x - x_min) / x_step)  # 경도 -> 열 인덱스
        row = int((y_max - y) / y_step)  # 위도 -> 행 인덱스 (y축은 상단에서 하단으로 감소)
        
        # 인덱스가 범위 내에 있는지 확인
        if 0 <= row < height and 0 <= col < width:
            grid_values[row, col] = value
        # else:
            # print(f'valuse in range: {row}, {col}') 

    
    # 이미지 데이터 생성 (RGBA)
    image_data = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if grid_values[i, j] == -9999:
                if is_point_off_range(grid_values, i, j, height, width):
                    # position of point is off the range
                    image_data[i][j] = [0, 0, 0, 0]
                    continue

            image_value = get_color_func[mode](grid_values[i, j])
            # 데이터 보간
            if image_value == [0, 0, 0 ,0]:
                left_value = image_data[i][j-1 if j > 1 else 0]
                up_value = image_data[i-1 if i > 1 else 0][j]
                image_value = np.mean([left_value,up_value], axis=0)
                if image_value[0] == 0 and image_value[1] == 0 and image_value[2] == 0:
                    image_value[3] = 0
                else:
                    image_value[3] = 255
            image_data[i][j] = image_value
            # image_data[i, j] = get_color_func[mode](grid_values[i, j])
    
    # 이미지를 PNG로 저장
    image = Image.fromarray(image_data, 'RGBA')
    image.save(output_path)
    print(f"Image saved to {output_path} with bounds: {bounds}")

def get_params_lc(file_path, var_name, grid_mapping):
  try :
    print('open file:', file_path)
    ds = nc.Dataset(file_path, format='NETCDF4')

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
  except Exception as e:
    print(e)

def latlon_from_lc(projAttrs: ProjectionAttributesLC, dim_x, dim_y):
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

def parseLc(step, dim_x, dim_y, attr_raw, projAttrs, conversion_array=None, use_index=0):
  print("start parseLc: prams =", step, dim_x, dim_y)
  lon_grid, lat_grid = latlon_from_lc(projAttrs, dim_x, dim_y)
  result = []
  for y in range(0, dim_y, step):
    for x in range(0, dim_x, step):
      attr_value = attr_raw[y, x]
      if conversion_array is not None:
        attr_value = conversion_array[attr_value][use_index]
      result.append([
          float(lon_grid[y, x]),  # 경도
          float(lat_grid[y, x]),  # 위도
          float(attr_value)  # 픽셀 값 (ushort -> int로 변환)
      ])
  return result



def get_params_geos(file_path, var_name, grid_mapping):
  # NetCDF 파일 열기
  ds = nc.Dataset(file_path, format='NETCDF4')

  # 이미지 픽셀 값과 차원 가져오기
  # ctt_raw = ds.variables['CTT'][:]  # CTT 데이터 (raw 값)
  attr_raw = ds.variables[var_name][:]  # CTT 데이터 (raw 값)
  dim_y, dim_x = attr_raw.shape
  print(f"{var_name} shape:", attr_raw.shape)

  # GEOS 투영 파라미터 (gk2a_imager_projection 변수에서 추출)
  projection = ds
  attr_map_key = 'from_global_attrs'
  if grid_mapping != None:
    projection = ds.variables[grid_mapping]
    attr_map_key = 'from_gk2a_imager_projection_attrs'

  attr_map = {
    'from_global_attrs': {
      'sub_lon': 'sub_longitude',
      'H': 'nominal_satellite_height',
      'a': 'earth_equatorial_radius',
      'b': 'earth_polar_radius',
      'cfac': 'cfac',
      'lfac': 'lfac',
      'coff': 'coff',
      'loff': 'loff',
    },
    'from_gk2a_imager_projection_attrs': {
      'sub_lon': 'longitude_of_projection_origin',
      'H': 'perspective_point_height',
      'a': 'semi_major_axis',
      'b': 'semi_minor_axis',
      'cfac': 'column_scale_factor',
      'lfac': 'line_scale_factor',
      'coff': 'column_offset',
      'loff': 'line_offset',
    }
  }
  sub_lon = projection.getncattr(attr_map[attr_map_key]['sub_lon'])
  if grid_mapping != None:
    sub_lon = projection.getncattr(attr_map[attr_map_key]['sub_lon']) * math.pi / 180.0  # 위성 경도, 도를 라디안으로 변환
  H = projection.getncattr(attr_map[attr_map_key]['H']) / 1000.0 # 위성 높이 (km 단위로 변환)
  a = projection.getncattr(attr_map[attr_map_key]['a'])          # 지구 적도 반지름 (미터)
  b = projection.getncattr(attr_map[attr_map_key]['b'])          # 지구 극 반지름 (미터)
  cfac = projection.getncattr(attr_map[attr_map_key]['cfac'])    # 열 방향 계수
  lfac = projection.getncattr(attr_map[attr_map_key]['lfac'])    # 행 방향 계수
  coff = projection.getncattr(attr_map[attr_map_key]['coff'])    # 열 중심
  loff = projection.getncattr(attr_map[attr_map_key]['loff'])    # 행 중심
  print(f"sub_lon (rad): {sub_lon}, H: {H} km, a: {a} m, b: {b} m, cfac: {cfac}, lfac: {lfac}, coff: {coff}, loff: {loff}")
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

def parseGeos(step, dim_x, dim_y, attr_raw, projAttrs, conversion_array=None, use_index=0):
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
            if conversion_array is not None:
              ctt_adjusted = conversion_array[ctt_adjusted][use_index]
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

def compress_file(input_filename, output_filename=None):
  """
  지정된 파일을 gzip으로 압축합니다.
  
  Args:
      input_filename (str): 원본 파일 경로 (예: 'data.json')
      output_filename (str, optional): 출력 파일 경로. 기본값은 input_filename + '.gz'
  """
  if output_filename is None:
    output_filename = input_filename + '.gz'
  
  with open(input_filename, 'rb') as f_in:
    with gzip.open(output_filename, 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)
  os.remove(input_filename)
  print(f"Compressed '{input_filename}' to '{output_filename}'")

def desctruct_att_lat_lon(attr_with_lat_lon):
  lons = [point[0] for point in attr_with_lat_lon]
  lats = [point[1] for point in attr_with_lat_lon]
  attr_values = [point[2] for point in attr_with_lat_lon]
  return lons, lats, attr_values

def print_result(lons, lats, attr_values, attr_to_get):
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

def mk_out_file_name(nc_file, step, out_dir):
  basename = Path(nc_file).stem
  # nc_coverage = basename.split('_')[1]
  nc_coverage = basename.split('_')[4][:2]
  nc_projection = basename.split('_')[4][-2:]
  out_file = f"{out_dir}/{basename}_step{step}.json"
  return out_file, nc_coverage, nc_projection

# GRID_MAPPING = {
#   "kma_grid": 'gk2a_imager_projection', # if data is ctps, grid_mapping = gk2a_imager_projection
#   "no_grid": None
# }

# get_params_func = {
#   'lc': get_params_lc,
#   'ge': get_params_geos
# }

# parse_func = {
#   'lc': parseLc,
#   'ge': parseGeos
# }

# if __name__ == '__main__' :
  # step = 10
  # out_dir = './'
  # parseResult = []


  # test ctps fd
  # attr_to_get = 'CTT'
  # nc_file = './working_script_samples/gk2a_ami_le2_ctps_fd020ge_202502170000.nc'
  # out_file, nc_coverage, nc_projection = mk_out_file_name(nc_file, step, out_dir)
  # attr_raw, dim_x, dim_y, projAttrs = get_params_geos(nc_file, attr_to_get, GRID_MAPPING.get('kma_grid', None))

  # test ctps lc
  # attr_to_get = 'CTT'
  # nc_file = './working_script_samples/ctps_ea_lc_202502170000.nc'
  # out_file, nc_coverage = mk_out_file_name(nc_file, step, out_dir)
  # attr_raw, dim_x, dim_y, projAttrs = get_params_lc(nc_file, attr_to_get, GRID_MAPPING.get('ctps', None))
  # parseResult = parseLc(step, dim_x, dim_y, attr_raw, projAttrs)

  # test ri105 ea
  # conversion_array = np.loadtxt('ir105_conversion_c.txt');
  # use_index = 0 # effective bright temperature : for colorIR``
  # use_index = 1 # effective bright temperature : for monoIR``
  # attr_to_get = 'image_pixel_values' # for ri105
  # nc_file = './working_script_samples/gk2a_ami_le1b_ir105_ea020lc_202502170000.nc'
  # out_file, nc_coverage, nc_projection = mk_out_file_name(nc_file, step, out_dir)
  # attr_raw, dim_x, dim_y, projAttrs = get_params_func[nc_projection](nc_file, attr_to_get, GRID_MAPPING.get('no_grid', None))

  # test ri105 fd
  # conversion_array = np.loadtxt('ir105_conversion_c.txt');
  # use_index = 0 # effective bright temperature : for colorIR``
  # use_index = 1 # effective bright temperature : for monoIR``
  # attr_to_get = 'image_pixel_values' # for ri105
  # nc_file = './working_script_samples/gk2a_ami_le1b_ir105_fd020ge_202502170000.nc'
  # out_file, nc_coverage, nc_projection = mk_out_file_name(nc_file, step, out_dir)
  # attr_raw, dim_x, dim_y, projAttrs = get_params_geos(nc_file, attr_to_get, GRID_MAPPING.get('no_grid', None))

  # test adps fd
  # attr_to_get = 'ADPS' # for ri105
  # nc_file = './working_script_samples/gk2a_ami_le2_adps_ea020lc_202503091450.nc'
  # out_file, nc_coverage, nc_projection = mk_out_file_name(nc_file, step, out_dir)
  # attr_raw, dim_x, dim_y, projAttrs = get_params_func[nc_projection](nc_file, attr_to_get, GRID_MAPPING.get('kma_grid', None))

  # common code 
  # parseResult = parse_func[nc_projection](step, dim_x, dim_y, attr_raw, projAttrs, conversion_array, use_index)
  # print("parse result Done:", len(parseResult))

  # validation code
  # lons, lats, attr_values = desctruct_att_lat_lon(parseResult);

  # print_result(lons, lats, attr_values, attr_to_get)
  # save_to_file(out_file, parseResult)
  # show_plot(lons, lats, attr_values, nc_coverage, f"visualization: {out_file}")

