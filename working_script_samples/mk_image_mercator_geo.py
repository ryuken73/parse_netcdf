import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import json
from pathlib import Path;
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer, CRS

# Web Mercator 투영 정의
web_mercator_crs = CRS.from_epsg("3857")  # EPSG:3857 (Web Mercator)
wgs84_crs = CRS.from_epsg("4326")  # EPSG:4326 (WGS84)
transformer_to_mercator = Transformer.from_crs(wgs84_crs, web_mercator_crs, always_xy=True)

# def generate_image_from_data(data, output_path, image_size=(1200, 1040), bounds=[76.811834, 11.308528, 175.188166, 53.303712]):
def generate_image_from_data(data, output_path, image_size=(600, 520), bounds=[60, -80, 180, 80]):
# def generate_image_from_data(data, output_path, image_size=(600, 520)):
    """
    주어진 [[lon, lat, value], ...] 데이터를 Web Mercator 투영을 반영하여 이미지로 변환.
    """
    print(image_size)
    # 경계 설정 (WGS84 좌표)
    lon_min, lat_min, lon_max, lat_max = bounds
    
    # 이미지 크기
    width, height = image_size  # 1200x1040 (원본 비율에 가까운 해상도)
    print(f'bounds:', bounds)
    
    # WGS84 경계 좌표를 Web Mercator로 변환
    x_min, y_max = transformer_to_mercator.transform(lon_min, lat_max)  # 좌상단
    x_max, y_min = transformer_to_mercator.transform(lon_max, lat_min)  # 우하단
    if x_max < 0 :
        print(f'x_max is negative. covert to positive:', x_max)
        x_max = 2 * 20037508.342789244 + x_max 
    print(f"x_min, x_max", x_min, x_max)
    print(f"y_min, y_max", y_min, y_max)
    
    # Web Mercator 좌표 간격 계산
    x_step = (x_max - x_min) / (width - 1)
    y_step = (y_max - y_min) / (height - 1)
    print(f"x_step", x_step)
    print(f"y_step", y_step)
    
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

    # 색상 매핑 (검정-흰색 보간)
    def get_color_from_value(value):
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
    
    # 이미지 데이터 생성 (RGBA)
    image_data = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # if i < 100:
            #     image_data[i, j] = [255, 255, 0, 255]
                # continue
            image_data[i, j] = get_color_from_value(grid_values[i, j])
            # if image_data[i, j][0] == 255:
                # print(f"fill red {j} {j}")
    
    # 이미지를 PNG로 저장
    image = Image.fromarray(image_data, 'RGBA')
    image.save(output_path)
    print(f"Image saved to {output_path} with bounds: {bounds}")
    
    return bounds

## griddata 없이 생성
# def generate_image_from_data(data, output_path, image_size=(600, 520), bounds=[76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447]):
# def generate_image_from_data(data, output_path, image_size=(600, 520), bounds=[76.811834, 11.308528, 175.188166, 53.3073712]):
#     """
#     주어진 [[lon, lat, value], ...] 데이터를 600x520 이미지로 변환.
#     griddata 없이 직접 매핑.
#     """
#     # 경계 설정
#     lon_min, lat_min, lon_max, lat_max = bounds
    
#     # 이미지 크기
#     width, height = image_size  # 600x520
    
#     # 위경도 간격 계산
#     lon_step = (lon_max - lon_min) / (width - 1)
#     lat_step = (lat_max - lat_min) / (height - 1)
    
#     # 2D 배열 초기화 (값 저장용)
#     grid_values = np.full((height, width), -9999, dtype=np.float32)
    
#     # 데이터를 2D 배열에 매핑
#     for lon, lat, value in data:
#         # 위경도를 픽셀 인덱스로 변환
#         col = int((lon - lon_min) / lon_step)  # 경도 -> 열 인덱스
#         row = int((lat_max - lat) / lat_step)  # 위도 -> 행 인덱스 (위도가 상단에서 하단으로 감소)
        
#         # 인덱스가 범위 내에 있는지 확인
#         if 0 <= row < height and 0 <= col < width:
#             grid_values[row, col] = value
    
#     # 색상 매핑 (검정-흰색 보간)
#     def get_color_from_value(value):
#         if value == -9999:
#             return [0, 0, 0, 0]  # 투명
#         t_min, t_max = -100, 30
#         ratio = min(max((value - t_min) / (t_max - t_min), 0), 1)
#         r = int(255 * (1 - ratio))
#         g = int(255 * (1 - ratio))
#         b = int(255 * (1 - ratio))
#         return [r, g, b, 255]  # RGBA
    
#     # 이미지 데이터 생성 (RGBA)
#     image_data = np.zeros((height, width, 4), dtype=np.uint8)
#     for i in range(height):
#         for j in range(width):
#             image_data[i, j] = get_color_from_value(grid_values[i, j])
    
#     # 이미지를 PNG로 저장
#     image = Image.fromarray(image_data, 'RGBA')
#     image.save(output_path)
    
#     return bounds

# steps = [4, 5, 8, 10]
steps = [1]
for step in steps:
#   data_file = f'gk2a_ami_le1b_ir105_ea020lc_202502281500_step{step}.json
  data_file='gk2a_ami_le1b_ir105_fd020ge_202503200840_202503201740_step2.json'
#   data_file='gk2a_ami_le1b_ir105_fd020ge_202503200840_202503201740_step4.json'
  print('generate png for datafile:', data_file)
  with open(data_file, 'r') as data_json:
    data = json.load(data_json)
    output_path = Path(data_file).stem + '.png'
    # bounds = generate_image_from_data(data, output_path, image_size=(600, 520))
    # bounds = generate_image_from_data(data, output_path)
    bounds = generate_image_from_data(data, output_path, image_size=(1200, 1024))
    print(f"Image bounds: {bounds}")