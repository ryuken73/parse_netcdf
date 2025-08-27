# -*- coding:utf8 -*-
import json
import numpy as np
import pandas as pd
import geopandas
from pyproj import CRS, Transformer
from scipy.interpolate import Rbf
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from configAWSColors import Colors

matplotlib.use('Agg')

MIN_LON, MAX_LON = 124.4, 131.6
MIN_LAT, MAX_LAT = 33, 38.6
## 보간 정밀도에 영향을 줌 (widthXheight으로 보간 포인트 생성)
## 값이 클수록 촘촘히 데이터를 보간해서 contour라인이 부드러워지지만 생성시간이 많이 걸림
## 생성된 이미지를 보고 튜닝하면 될 듯(기본 200X200)
IMAGE_WIDTH_PIXELS = 200 
IMAGE_HEIGHT_PIXELS = 200
EPSILON = 0.1

def create_color_map(boundaries, colors_list, bottom_value, top_value, show_preview=True):
  # Matplotlib ListedColormap을 사용하여 커스텀 컬러맵 생성
  # N = len(boundaries)이므로, boundaries의 개수와 colors_list의 개수를 맞춰야 함.
  cmap = mcolors.ListedColormap(colors_list)

  # 경계값에 따라 Colormap을 정규화하는 BoundaryNorm 생성
  # norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
  norm = mcolors.BoundaryNorm(boundaries, cmap.N, extend=True)

  # 0.1보다 작은 값과 100보다 큰 값에 대한 색상 설정
  # 이미지에서 0.1 미만은 F0F0F0, 100 초과는 242424로 보임
  cmap.set_under(bottom_value) # 0.1 미만 값
  cmap.set_over(top_value) # 100 초과 값

  if show_preview:
    # 시각화 예시
    # 임의의 데이터 생성
    data = np.random.uniform(4, 6, size=(10, 10))

    # Matplotlib를 사용하여 plot
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(data, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax, boundaries=boundaries, extend='both',
                ticks=boundaries, spacing='proportional')
    ax.set_title("Custom Colormap from User-provided Images")

    plt.show()
  return cmap, norm

def load_json(json_file):
  try :
    with open(json_file, encoding='utf-8') as f:
      data = json.load(f)
      return data
  except Exception as e:
    print(f'error to open json file {json_file}')

def is_valid_stn_data(stn_data):
    # 'STN_NAME' 키가 없거나 값이 None이면 False 반환
    stn_name = stn_data.get('STN_NAME')
    if stn_name is None:
        return False

    # 값이 문자열 타입이 아니면 False 반환
    if not isinstance(stn_name, str):
        return False

    # 값이 'test'로 시작하면 False, 아니면 True 반환
    return not stn_name.startswith('test')

def filter_json(json_data, pick_column, invalid_values, value_to_multiply):
  try :
    filter_only_valid = [
      stn_data for stn_data 
      in json_data 
      if (is_valid_stn_data(stn_data)) and (stn_data[pick_column] not in invalid_values)
    ]
    filter_only_column = [
      {
        "stn_name": stn_data["STN_NAME"],
        "lat": stn_data["LAT"],
        "lon": stn_data["LON"],
        pick_column: stn_data[pick_column] * value_to_multiply 
      } for stn_data in filter_only_valid
    ]
    return filter_only_column
  except Exception as e:
    print(e)
    print(f'error to extract key {pick_column}')

def is_in_boundary(station_data):
  lon = float(station_data['lon'])
  lat = float(station_data['lat'])
  return MIN_LON <= lon <= MAX_LON and MIN_LAT <= lat <= MAX_LAT

# --- 1. 좌표변환 함수(station data) ---
def transform_data(transformer, data_to_transform, column_to_pick):
  """
    json array에서 위/경도, 관심데이터를 numpy list로 만들고
    위/경도 좌표는 Web Mercator로 변환한다.
  """
  data = list(filter(is_in_boundary, data_to_transform));
  lons = np.array([float(item["lon"]) for item in data])
  lats = np.array([float(item["lat"]) for item in data])
  aws_values = np.array([float(item[column_to_pick]) for item in data])
  station_xs_wm, station_ys_wm = transformer.transform(lons, lats)
  return lons, lats, aws_values, station_xs_wm, station_ys_wm

# --- 2. 좌표변환(보간데이터를 만들 Grid 꼭지점 4개)
def transform_image_boundary_wm(transformer, min_lon, max_lon, min_lat, max_lat):
  wm_coords_top_left = transformer.transform(min_lon, max_lat)
  wm_coords_top_right = transformer.transform(max_lon, max_lat)
  wm_coords_bottom_left = transformer.transform(min_lon, min_lat)

  min_x_wm = wm_coords_top_left[0]
  max_x_wm = wm_coords_top_right[0]
  min_y_wm = wm_coords_bottom_left[1]
  max_y_wm = wm_coords_top_left[1]
  return min_x_wm, max_x_wm, min_y_wm, max_y_wm

# --- 3. 보간 수행 함수 ---
def perform_interpolation(station_xs_wm, station_ys_wm, snow_depths,
                          min_x_wm, max_x_wm, min_y_wm, max_y_wm,
                          image_width_pixels, image_height_pixels, epsilon=0.1):
  """
  Web Mercator 좌표에서 RBF 보간을 수행하고, 보간된 그리드 데이터를 반환합니다.
  """
  xi_wm = np.linspace(min_x_wm, max_x_wm, image_width_pixels)
  yi_wm = np.linspace(min_y_wm, max_y_wm, image_height_pixels)
  XI_WM, YI_WM = np.meshgrid(xi_wm, yi_wm)

  rbf = Rbf(station_xs_wm, station_ys_wm, snow_depths, function='linear', epsilon=epsilon, smooth=0)
  ZI_WM = rbf(XI_WM, YI_WM)

  return XI_WM, YI_WM, ZI_WM

def create_kor_boundary_mask(mask, korea_geojson_path: str, XI_WM, YI_WM, lat_cutoff):
    try:
        korea_boundary = geopandas.read_file(korea_geojson_path)
        print('korea boundary GeoJSON loaded successfully.')

        korea_boundary_wm = korea_boundary.to_crs(epsg=3857)

        grid_points_df = pd.DataFrame({
            'x': XI_WM.ravel(),
            'y': YI_WM.ravel(),
            'index': range(len(XI_WM.ravel()))
        })
        grid_points_gdf = geopandas.GeoDataFrame(
            grid_points_df,
            geometry=geopandas.points_from_xy(grid_points_df['x'], grid_points_df['y']),
            crs="EPSG:3857"
        )

        transformer = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:3857"), always_xy=True)
        # lat_cutoff = 37.7
        _, y_cutoff_wm = transformer.transform(127, lat_cutoff)
        # 3. 위도 기준 필터링
        # Web Mercator의 Y 좌표가 강화도 기준보다 북쪽에 있는 점들만 추출
        points_to_check_gdf = grid_points_gdf[grid_points_gdf['y'] >= y_cutoff_wm].copy()

        korea_boundary_union = korea_boundary_wm.geometry.unary_union
        points_within_korea = points_to_check_gdf.geometry.within(korea_boundary_union)

        # mask = np.zeros_like(ZI_WM, dtype=bool)
        mask_indices = points_to_check_gdf[~points_within_korea]['index'].values
        mask.ravel()[mask_indices] = True
        return mask
    except Exception as e:
        print(f"Error loading or processing GeoJSON: {e}")
        print("Proceeding without boundary masking. The output will show data outside Korea.")
        return mask  

# --- 4. 마스크 적용 ---
def apply_mask(XI_WM, YI_WM, ZI_WM, korea_geojson_path):
  try:
    # 대한민국 바운더리 마스킹 적용
    # lat_cutoff = 37.7
    lat_cutoff = 33
    mask = np.zeros_like(ZI_WM, dtype=bool)
    mask_boundary = create_kor_boundary_mask(mask, korea_geojson_path, XI_WM, YI_WM, lat_cutoff)

    # 음수 또는 아주 작은 값 마스킹 (0.05 미만)
    mask_boundary[ZI_WM < 0.05] = True

    Z_masked = np.ma.array(ZI_WM, mask=mask_boundary)
    return Z_masked
    # print(f'Number of grid points checked: {len(points_to_check_gdf)} out of {len(grid_points_gdf)}.')
  except Exception as e:
    print(f"Error loading or processing GeoJSON: {e}")
    print("Proceeding without boundary masking. The output will show data outside Korea.")
    mask = np.zeros_like(ZI_WM, dtype=bool)
    mask[ZI_WM < 0.05] = True
    Z_masked = np.ma.array(ZI_WM, mask=mask)
    return Z_masked
  
def save_contour_image(XI_WM, YI_WM, Z_masked, cmap, norm, station_xs_wm, 
                       station_ys_wm, aws_values, lons, out_image_name, print_values_min):
  fig = plt.figure(figsize=(25, 24), dpi=100)
  size = fig.get_size_inches() * fig.dpi
  print('Figure size:', size)
  ax = plt.gca()
  ax.contourf(XI_WM, YI_WM, Z_masked, levels=50, cmap=cmap, norm=norm, extend='neither', alpha=1)
  # contour = ax.contourf(XI_WM, YI_WM, Z_masked, levels=50, cmap=cmap, extend='neither', alpha=1)

  ax.set_axis_off()
  plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
  plt.margins(0,0)

  for i in range(len(lons)):
    if aws_values[i] > print_values_min:
      ax.text(station_xs_wm[i], station_ys_wm[i], f'{aws_values[i]:.1f}',
        ha='center', va='center', color='red', fontsize=12)

  fig.savefig(out_image_name, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='none')
  plt.close(fig)
  print(f"Image {out_image_name} generated.")


# Sample code
# 입력 aws_file, 한국 경계선 geojson파일 필요

'''
aws_file = r'D:/002.Code/002.node/weather_api/kma_fetch/data/weather/aws/2025-08-05/AWS_MIN_202508052356.json'
korea_boundary_geojson = 'skorea-provinces-2018-geo.json'

# 1. 데이터 로드
json_data = load_json(aws_file)

column_to_pick = 'RN_15M'
invalid_value = -999
value_to_multiply = 0.1
picked_data = filter_json(json_data, column_to_pick, invalid_value)
# print(picked_data)

# 2. 데이터 변환(좌표 변환)
transformer = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:3857"), always_xy=True)
lons, lats, aws_values, station_xs_wm, station_ys_wm = transform_data(transformer, picked_data, column_to_pick)

# 3. 이미지를 생성할 그리드 범위를 구한다
min_x_wm, max_x_wm, min_y_wm, max_y_wm = transform_image_boundary_wm(transformer, MIN_LON, MAX_LON, MIN_LAT, MAX_LAT)

# 4. 데이터 보간
XI_WM, YI_WM, ZI_WM = perform_interpolation(
    station_xs_wm, station_ys_wm, aws_values,
    min_x_wm, max_x_wm, min_y_wm, max_y_wm,
    IMAGE_WIDTH_PIXELS, IMAGE_HEIGHT_PIXELS, EPSILON
)

# X. create color map
boundaries = Colors[column_to_pick]["boundaries"]
colors_list = Colors[column_to_pick]["colors_list"]
bottom_value = Colors[column_to_pick]["bottom_value"]
top_value = Colors[column_to_pick]["top_value"]

z_masked = apply_mask(XI_WM, YI_WM, ZI_WM, korea_boundary_geojson)
cmap, norm = create_color_map(boundaries, colors_list, bottom_value, top_value, False)
save_contour_image(XI_WM, YI_WM, z_masked, cmap, norm, station_xs_wm, station_ys_wm, aws_values, lons, 'a.png')
'''