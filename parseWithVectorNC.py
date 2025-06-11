# vector 연산을 기반으로 이미지를 빠르게 만드는 모듈
# 보간은 없음 (보간 로직 만들기 어려움)

import numpy as np
from pyproj import Proj
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from PIL import Image
from pyproj import Transformer, CRS

# Web Mercator 투영 정의
web_mercator_crs = CRS.from_epsg("3857")
wgs84_crs = CRS.from_epsg("4326")
transformer_to_mercator = Transformer.from_crs(wgs84_crs, web_mercator_crs, always_xy=True)

LAT_LON_NC_FILE = './assets/gk2a_ami_fd020ge_latlon.nc'

# get_color_from_temperature 함수 (기존과 동일)
def get_color_from_temperature(temp):
    colors = {
        20: [0, 0, 0], -20: [255, 255, 255], -21: [135, 206, 235], -30: [0, 0, 255],
        -40: [0, 255, 0], -45: [144, 238, 144], -50: [255, 255, 0], -60: [255, 0, 0],
        -70: [0, 0, 0], -80: [255, 255, 255], -81: [128, 128, 128], -90: [128, 0, 128]
    }
    if temp > 20 or temp < -90:
        return [0, 0, 0, 0]
    temp = max(-90, min(20, temp))
    keys = sorted(colors.keys(), reverse=True)
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
            return [0, 0, 0, 255]
        else:
            start_temp, end_temp = keys[0], keys[1]
            start_color, end_color = colors[start_temp], colors[end_temp]
    if start_temp == end_temp or (start_temp in [-20, -80] and temp == start_temp + 1):
        return end_color + [255]
    ratio = (start_temp - temp) / (start_temp - end_temp)
    r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
    g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
    b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
    alpha = (20 - temp) * (255 / 40) if temp > -20 else 255
    return [r, g, b, alpha]

def get_mono_color_from_temperature(value, factor=5):
    if value == -9999:
        return [0, 0, 0, 0]  # 투명
    
    t_min, t_max = -100, 30
    ratio = min(max((value - t_min) / (t_max - t_min), 0), 1)
    
    # 기본 밝기 계산 (원래 함수와 동일)
    base_r = (1 - ratio)  # 0.0 ~ 1.0
    base_g = (1 - ratio)
    base_b = (1 - ratio)
    
    # 밝기 조정: factor에 따라 밝기를 증폭 (1~10)
    # factor가 1일 때는 원래 밝기, 10일 때는 최대 밝기에 가깝게
    brightness_boost = 1 + (factor - 1) * 0.2  # factor 1 -> 1.0, factor 10 -> 2.8
    r = int(min(255 * base_r * brightness_boost, 255))
    g = int(min(255 * base_g * brightness_boost, 255))
    b = int(min(255 * base_b * brightness_boost, 255))
    alpha = int(min(255 * (base_r+base_g+base_b)/3 * brightness_boost, 255))
    # alpha = int(min(255 * (r+g+b)/3 * brightness_boost, 255))
    return [r, g, b, alpha]  # RGBA

def get_mono_color_from_temperature_old(value):
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

def generate_image_from_data_fast(data, output_path, image_size=(600, 520), bounds=[60, -80, 180, 80], color_mode="gray"):
    """
    NumPy 배열 [[lon, lat, value], ...] 데이터를 Web Mercator 투영으로 이미지 변환.
    """
    # 경계 설정
    lon_min, lat_min, lon_max, lat_max = bounds
    width, height = image_size

    # 데이터 분리
    lons = data[:, 0]
    lats = data[:, 1]
    values = data[:, 2]

    # 경도 180도를 넘는 경우 처리: Web Mercator 투영을 위해 -180~180도로 변환
    lons_for_transform = np.where(lons > 180, lons - 360, lons)

    # Web Mercator로 변환
    x, y = transformer_to_mercator.transform(lons_for_transform, lats)

    # bounds의 경계 좌표도 변환
    x_min, y_max = transformer_to_mercator.transform(lon_min if lon_min <= 180 else lon_min - 360, lat_max)
    x_max, y_min = transformer_to_mercator.transform(lon_max if lon_max <= 180 else lon_max - 360, lat_min)

    # 경도 180도를 넘는 경우 x 좌표 조정
    # Web Mercator에서 180도를 넘는 경도는 x 좌표가 음수로 나타날 수 있으므로, 이를 양수로 변환
    x = np.where(lons > 180, x + 2 * 20037508.342789244, x)  # 180도를 넘는 경우 x 좌표를 이어 붙임
    if lon_max > 180:
        x_max += 2 * 20037508.342789244  # bounds의 x_max도 조정

    print(f"x_min, x_max:", x_min, x_max)
    print(f"y_min, y_max:", y_min, y_max)

    # 픽셀 간격 계산
    x_step = (x_max - x_min) / (width - 1)
    y_step = (y_max - y_min) / (height - 1)
    print(f"x_step:", x_step)
    print(f"y_step:", y_step)

    # 이미지 좌표로 매핑
    cols = np.clip(((x - x_min) / x_step).astype(int), 0, width - 1)
    rows = np.clip(((y_max - y) / y_step).astype(int), 0, height - 1)

    # 2D 그리드 초기화
    grid_values = np.full((height, width), -9999, dtype=np.float32)
    grid_values[rows, cols] = values  # 값 매핑

    print('grid_values sample:', grid_values[:5])
    print("Image pixel values 범위:", np.min(grid_values), "to", np.max(grid_values))

    # 색상 매핑을 위한 벡터화 함수
    def apply_color_mapping(values):
        colors = np.zeros((height, width, 4), dtype=np.uint8)
        flat_values = values.flatten()
        global flat_colors
        if color_mode == 'gray': 
          flat_colors = np.array([get_mono_color_from_temperature(val) if val != -9999 else [0, 0, 0, 0] 
                                for val in flat_values]) 
        else: 
          flat_colors = np.array([get_color_from_temperature(val) if val != -9999 else [0, 0, 0, 0] 
                                for val in flat_values])
        colors = flat_colors.reshape(height, width, 4)
        return colors

    # 이미지 데이터 생성
    image_data = apply_color_mapping(grid_values)
    print('length of image_dat:', len(image_data))
    print('sample of image_dat:', image_data[:100])
    print('sample of image_dat:', image_data[600:800])
    print('shape and size of image_dat:', image_data.shape, image_data.size, image_data.dtype)

    # PNG 저장
    image = Image.fromarray(image_data.astype(np.uint8), 'RGBA')
    image.save(output_path)
    print(f"Image saved to {output_path} with bounds: {bounds}")
    return bounds

def load_conversion_table(file_path):
    conversion_table = []
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 2:
                conversion_table.append(float(values[1]))
    return np.array(conversion_table)

# get ir105_fd lon, lat, values from nc
def read_ir105_fd_fast_with_vector (file_path, step, attr_to_get, conversion_file):
    ds = nc.Dataset(file_path, format='NETCDF4')
    image_pixel_values = ds.variables[attr_to_get][:]
    dim_y, dim_x = image_pixel_values.shape
    print("image_pixel_values shape:", image_pixel_values.shape)

    # 제공된 NetCDF 파일에서 위경도 테이블 로드
    latlon_ds = nc.Dataset(LAT_LON_NC_FILE, format='NETCDF4')
    lat = latlon_ds.variables['lat'][:]
    lon = latlon_ds.variables['lon'][:]
    latlon_ds.close()

    # step을 적용한 인덱스 생성
    y_indices = np.arange(0, dim_y, step)
    x_indices = np.arange(0, dim_x, step)
    Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')

    # 인덱스를 사용해 위경도 배열 샘플링
    sampled_lat = lat[Y, X]
    sampled_lon = lon[Y, X]

    # 경도 변환: -180~0도 사이 값을 180~360도로 변환
    valid_lon_mask = (sampled_lon >= -180) & (sampled_lon <= 180)  # 유효한 경도 값만 선택
    adjusted_lon = np.where(valid_lon_mask, sampled_lon, np.nan)  # 비정상 값은 NaN으로 처리
    adjusted_lon = np.where(adjusted_lon < 0, adjusted_lon + 360, adjusted_lon)  # -180~0도를 180~360도로 변환

    # 위경도 범위 마스크 적용: 경도 범위를 50~230도로 확장
    latlon_mask = (sampled_lat >= -80) & (sampled_lat <= 80) & (adjusted_lon >= 30) & (adjusted_lon <= 230)
    final_mask = latlon_mask

    #   sub_lon = ds.getncattr('sub_longitude')
    #   H = ds.getncattr('nominal_satellite_height')
    #   a = ds.getncattr('earth_equatorial_radius')
    #   b = ds.getncattr('earth_polar_radius')
    #   cfac = ds.getncattr('cfac')
    #   lfac = ds.getncattr('lfac')
    #   coff = ds.getncattr('coff')
    #   loff = ds.getncattr('loff')

    #   y_indices = np.arange(0, dim_y, step)
    #   x_indices = np.arange(0, dim_x, step)
    #   Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
    #   degtorad = np.pi / 180.0
    #   x = degtorad * ((X - coff) * 2**16 / cfac)
    #   y = degtorad * ((dim_y - Y - 1 - loff) * 2**16 / lfac)
    #   cos_x = np.cos(x)
    #   cos_y = np.cos(y)
    #   sin_y = np.sin(y)
    #   Sd = np.sqrt((42164.0 * cos_x * cos_y)**2 - (cos_y**2 + 1.006739501 * sin_y**2) * 1737122264)
    #   valid_mask = Sd >= 0
    #   Sn = np.where(valid_mask, (42164.0 * cos_x * cos_y - Sd) / (cos_y**2 + 1.006739501 * sin_y**2), np.nan)
    #   S1 = 42164.0 - (Sn * cos_x * cos_y)
    #   S2 = Sn * (np.sin(x) * cos_y)
    #   S3 = -Sn * sin_y
    #   Sxy = np.sqrt(S1**2 + S2**2)
    #   lon = np.where(valid_mask, (np.arctan2(S2, S1) + sub_lon) / degtorad, np.nan)
    #   lat = np.where(valid_mask, np.arctan2(1.006739501 * S3, Sxy) / degtorad, np.nan)
    #   # latlon_mask = (lat >= -80) & (lat <= 80) & (lon >= -180) & (lon <= 190)
    #   latlon_mask = (lat >= -80) & (lat <= 80) & (lon >= 60) & (lon <= 180)
    #   final_mask = valid_mask & latlon_mask

    # 변환 테이블 적용
    values = image_pixel_values[Y, X] 
    lut = load_conversion_table(conversion_file)

    # 마스크를 사용한 안전한 변환
    mask = (values >= 0) & (values < len(lut))
    converted_values = np.full_like(values, -9999, dtype=float)  # 기본값 -9999
    converted_values[mask] = lut[values[mask]]  # 유효한 값만 변환

    print("Max value in converted_values:", np.max(converted_values))
    print("Min value in converted_values:", np.min(converted_values))

    # result에 converted_values 사용
    result = np.column_stack([
        adjusted_lon[final_mask].astype(float),
        sampled_lat[final_mask].astype(float),
        converted_values[final_mask].astype(float)
    ])

    # 디버깅 출력
    lons = result[:, 0]
    lats = result[:, 1]
    print("Longitude 범위:", np.min(lons) if lons.size > 0 else "No valid points", 
            "to", np.max(lons) if lons.size > 0 else "No valid points")
    print("Latitude 범위:", np.min(lats) if lats.size > 0 else "No valid points", 
            "to", np.max(lats) if lats.size > 0 else "No valid points")
    print("샘플 데이터:", result[:5])

    ds.close()

    return result

# get ir105_ea lon, lat, values from nc
def read_ir105_ea_fast_with_vector(file_path, step, attr_to_get, conversion_file):
  # 파일 열기
  print('execute nc.Dataset')
  ds = nc.Dataset(file_path, 'r')

  # 이미지 픽셀 값과 차원 가져오기
  image_pixel_values = ds.variables[attr_to_get][:]
  dim_y, dim_x = image_pixel_values.shape
  print("image_pixel_values shape:", image_pixel_values.shape)

  # 투영 정보 가져오기 (LCC)
  proj_attrs = ds.__dict__
  central_meridian = proj_attrs["central_meridian"]  # 126.0
  standard_parallel1 = proj_attrs["standard_parallel1"]  # 30.0
  standard_parallel2 = proj_attrs["standard_parallel2"]  # 60.0
  origin_latitude = proj_attrs["origin_latitude"]  # 38.0
  false_easting = proj_attrs["false_easting"]  # 0.0
  false_northing = proj_attrs["false_northing"]  # 0.0
  pixel_size = proj_attrs["pixel_size"]  # 2000.0 (미터 단위)
  upper_left_easting = proj_attrs["upper_left_easting"]  # -2999000.0
  upper_left_northing = proj_attrs["upper_left_northing"]  # 2599000.0

  # LCC 투영 정의
  proj = Proj(
      proj="lcc",
      lat_1=standard_parallel1,
      lat_2=standard_parallel2,
      lat_0=origin_latitude,
      lon_0=central_meridian,
      x_0=false_easting,
      y_0=false_northing,
      units="m",
      ellps="WGS84"
  )

  # 각 픽셀의 easting, northing 좌표 계산 (이미 벡터화됨)
  easting = np.linspace(upper_left_easting, upper_left_easting + pixel_size * (dim_x - 1), dim_x)
  northing = np.linspace(upper_left_northing, upper_left_northing - pixel_size * (dim_y - 1), dim_y)
  easting_grid, northing_grid = np.meshgrid(easting, northing)

  # LCC -> WGS84 변환 (벡터화)
  lon_grid, lat_grid = proj(easting_grid, northing_grid, inverse=True)

  # 샘플링 인덱스 생성
  step = 1
  y_indices = np.arange(0, dim_y, step)
  x_indices = np.arange(0, dim_x, step)
  Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')

  # 벡터화된 데이터 추출
  lons = lon_grid[Y, X].astype(float)  # 경도
  lats = lat_grid[Y, X].astype(float)  # 위도
  values = image_pixel_values[Y, X].astype(int)  # 픽셀 값

  lut = load_conversion_table(conversion_file)

  # 마스크를 사용한 안전한 변환
  mask = (values >= 0) & (values < len(lut))
  converted_values = np.full_like(values, -9999, dtype=float)  # 기본값 -9999
  converted_values[mask] = lut[values[mask]]  # 유효한 값만 변환

  print("Max value in converted_values:", np.max(converted_values))
  print("Min value in converted_values:", np.min(converted_values))


  # 결과 배열 생성
  result = np.column_stack([lons.flatten(), lats.flatten(), converted_values.flatten()])

  # 위경도 및 픽셀 값 범위 출력 (디버깅용)
  print("Longitude 범위:", np.min(lons) if lons.size > 0 else "No valid points",
        "to", np.max(lons) if lons.size > 0 else "No valid points")
  print("Latitude 범위:", np.min(lats) if lats.size > 0 else "No valid points",
        "to", np.max(lats) if lats.size > 0 else "No valid points")
  print("Image pixel values 범위:", np.min(values) if values.size > 0 else "No valid points",
        "to", np.max(values) if values.size > 0 else "No valid points")

  # 결과 확인 (처음 5개만 출력)
  print("샘플 데이터:", result[:5])

  # 파일 닫기
  ds.close()

  return result

def get_nc_coverage(nc_file):
  basename = Path(nc_file).stem
  return basename.split('_')[4][:2]

def get_json_fname(out_dir, nc_file, step):
  basename = Path(nc_file).stem
  out_file = f"{out_dir}/{basename}_step{step}.json"
  return out_file

def mk_out_file_name(nc_file, step, out_dir):
  basename = Path(nc_file).stem
  # nc_coverage = basename.split('_')[1]
  nc_coverage = basename.split('_')[4][:2]
  nc_projection = basename.split('_')[4][-2:]
  out_file = f"{out_dir}/{basename}_step{step}.json"
  return out_file, nc_coverage, nc_projection

def resize_image(src_image_path, target_image_path, target_resolution):
  with Image.open(src_image_path) as src_img:
    resized = src_img.resize(target_resolution, Image.Resampling.LANCZOS)
    resized.save(target_image_path)



# 사용 예시
# output_path = "output_image.png"
# bounds = [60, -80, 180, 80]
# generate_image_from_data(result, output_path, image_size=(600, 520), bounds=bounds)