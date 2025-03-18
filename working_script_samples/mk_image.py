import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import json
from pathlib import Path;
import cartopy.crs as ccrs
import cartopy.feature as cfeature


## griddata 없이 생성
# def generate_image_from_data(data, output_path, image_size=(600, 520), bounds=[76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447]):
def generate_image_from_data(data, output_path, image_size=(300, 260), bounds=[76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447]):
# def generate_image_from_data(data, output_path, image_size=(600, 520), bounds=[76.811834, 11.308528, 175.188166, 53.3073712]):
    """
    주어진 [[lon, lat, value], ...] 데이터를 600x520 이미지로 변환.
    griddata 없이 직접 매핑.
    """
    # 경계 설정
    lon_min, lat_min, lon_max, lat_max = bounds
    
    # 이미지 크기
    width, height = image_size  # 600x520
    
    # 위경도 간격 계산
    lon_step = (lon_max - lon_min) / (width - 1)
    lat_step = (lat_max - lat_min) / (height - 1)
    
    # 2D 배열 초기화 (값 저장용)
    grid_values = np.full((height, width), -9999, dtype=np.float32)
    
    processed = 0
    dup_count = 0
    # 데이터를 2D 배열에 매핑
    for lon, lat, value in data:
        # 위경도를 픽셀 인덱스로 변환
        col = int((lon - lon_min) / lon_step)  # 경도 -> 열 인덱스
        row = int((lat_max - lat) / lat_step)  # 위도 -> 행 인덱스 (위도가 상단에서 하단으로 감소)
        processed += 1
        # print(col, row, value, processed)
        
        # 인덱스가 범위 내에 있는지 확인
        if 0 <= row < height and 0 <= col < width:
            if grid_values[row, col] != -9999:
                dup_count += 1
                print(f"dup assign value to", grid_values[row, col], row, col, processed, dup_count)
            grid_values[row, col] = value

    
    # 색상 매핑 (검정-흰색 보간)
    def get_color_from_value(value):
        if value == -9999:
            return [0, 0, 0, 0]  # 투명
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
            image_data[i, j] = get_color_from_value(grid_values[i, j])
    
    # 이미지를 PNG로 저장
    image = Image.fromarray(image_data, 'RGBA')
    image.save(output_path)
    
    return bounds

# steps = [4, 5, 8, 10]
steps = [4]
for step in steps:
  data_file = f'gk2a_ami_le1b_ir105_ea020lc_202502281500_step{step}.json'
  print('generate png for datafile:', data_file)
  with open(data_file, 'r') as data_json:
    data = json.load(data_json)
    output_path = Path(data_file).stem + '.png'
    # bounds = generate_image_from_data(data, output_path, image_size=(600, 520))
    bounds = generate_image_from_data(data, output_path, image_size=(750, 650))
    # bounds = generate_image_from_data(data, output_path, image_size=(1024, 768))
    print(f"Image bounds: {bounds}")