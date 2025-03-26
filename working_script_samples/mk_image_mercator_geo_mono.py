import os
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

# get_color_from_temperature 함수
def get_color_from_temperature(temp):
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
    
    # Clamp temperature to valid range (20 to -90)
    if temp > 20 or temp < -90 :
        return [0, 0, 0, 0]
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
    print('grid_values sample:', grid_values[:5])

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
            if grid_values[i, j] == -9999:
                if is_point_off_range(grid_values, i, j, height, width):
                    # position of point is off the range
                    image_data[i][j] = [0, 0, 0, 0]
                    continue
            image_value = get_color_from_value(grid_values[i, j])
            # image_value = get_color_from_temperature(grid_values[i, j])
            if image_value == [0, 0, 0 ,0]:
                left_value = image_data[i][j-1 if j > 1 else 0]
                up_value = image_data[i-1 if i > 1 else 0][j]
                # print("left, right, up, down", left_value, right_value, up_value, down_value)
                image_value = np.mean([left_value,up_value], axis=0)
                if image_value[0] == 0 and image_value[1] == 0 and image_value[2] == 0:
                    image_value[3] = 0
                else:
                    image_value[3] = 255
            image_data[i][j] = image_value
            # image_data[i, j] = get_color_from_value(grid_values[i, j])
    
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
out_dir = r'D:\002.Code\099.study\deck.gl.test\public'
steps = [1]
for step in steps:
#   data_file = f'gk2a_ami_le1b_ir105_ea020lc_202502281500_step{step}.json
#   data_file='gk2a_ami_le1b_ir105_ea020lc_202503231410_202503232310_step1.json'
#   data_file='gk2a_ami_le1b_ir105_fd020ge_202503200840_202503201740_step2.json'
#   data_file='gk2a_ami_le1b_ir105_fd020ge_202503200840_202503201740_step4.json'
  data_file='gk2a_ami_le1b_ir105_fd020ge_202503231250_202503232150_step3.json'
  print('generate png for datafile:', data_file)
  with open(data_file, 'r') as data_json:
    data = json.load(data_json)
    # output_path = Path(data_file).stem + '.png'
    
    output_path = Path(data_file).stem + '.png'
    # bounds = generate_image_from_data(data, output_path, image_size=(600, 520))
    # bounds = generate_image_from_data(data, output_path)
    bounds = generate_image_from_data(data, output_path, image_size=(1200, 1024))
    # bounds = generate_image_from_data(data, output_path, image_size=(2048, 2048))
    # bounds = generate_image_from_data(data, output_path, image_size=(3192, 3192))
    # bounds = generate_image_from_data(data, output_path, image_size=(4096, 4096))
    # bounds = generate_image_from_data(data, output_path, image_size=(1500, 1500))
    # bounds = generate_image_from_data(data, output_path, image_size=(1024, 768))
    print(f"Image bounds: {bounds}")