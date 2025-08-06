import os
from pathlib import Path
from parseAWSJson import *
from watchFolder_Thread import start_watching
from config import get_config
from PIL import Image
import sys
import multiprocessing

config = get_config()
print(f'Running in {config.ENV} mode')
print(f'OUT_PATH_AWS = {config.OUT_PATH_AWS}')
print(f'WATCH_PATH_AWS = {config.WATCH_PATH_AWS}')

MIN_LON, MAX_LON = 124.4, 131.6
MIN_LAT, MAX_LAT = 33, 38.6
## 보간 정밀도에 영향을 줌 (widthXheight으로 보간 포인트 생성)
## 값이 클수록 촘촘히 데이터를 보간해서 contour라인이 부드러워지지만 생성시간이 많이 걸림
## 생성된 이미지를 보고 튜닝하면 될 듯(기본 200X200)
IMAGE_WIDTH_PIXELS = 200 
IMAGE_HEIGHT_PIXELS = 200
EPSILON = 0.1

korea_boundary_geojson = 'skorea-provinces-2018-geo.json'

VALUE_TO_MULTIPLY = {
  "RN_15M": 0.1
}

SAVE_IMAGE_STEPS =  [1, 5, 10]
IMAGE_SIZE = {
  5: (1703, 1956),
  10: (1277, 1467)
}

out_dir = config.OUT_PATH_AWS
highest_step = 1

def callback(aws_file):
  print(f'processing file {aws_file}')
  sub_dir = os.path.dirname(aws_file).split(config.OS_SEP)[-1:][0]
  print(sub_dir)
  save_dir = f"{out_dir}/{sub_dir}"
  os.makedirs(save_dir, exist_ok=True)
  print(f'save file to {save_dir}')
  print(f'start read AWS json file')
  aws_image_name = f'{save_dir}/{Path(aws_file).stem}_step{highest_step}.png'
  print('start save image')

  try :
    # 1. 데이터 로드
    json_data = load_json(aws_file)

    column_to_pick = 'RN_15M'
    invalid_value = -999
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

    # 5. create color map (Colors는 configAWSColors에 정의가 되어있다.)
    boundaries = Colors[column_to_pick]["boundaries"]
    colors_list = Colors[column_to_pick]["colors_list"]
    bottom_value = Colors[column_to_pick]["bottom_value"]
    top_value = Colors[column_to_pick]["top_value"]

    # 6. 대한민국 경계로 masking하고 값이 너무 작은 경우도 투명처리한다.
    z_masked = apply_mask(XI_WM, YI_WM, ZI_WM, korea_boundary_geojson)

    # 7. color map을 만든다.
    cmap, norm = create_color_map(boundaries, colors_list, bottom_value, top_value, False)

    # 8. contour이미지를 만들고 저장한다.
    save_contour_image(XI_WM, YI_WM, z_masked, cmap, norm, station_xs_wm, station_ys_wm, aws_values, lons, aws_image_name)
    print(f'done save image {aws_image_name}')
    print("waiting for next files...")
  except Exception as e:
    print('error', e)
    print("waiting for next files...")

if __name__ == '__main__' :
  start_watching(config.WATCH_PATH_AWS, None, callback)
