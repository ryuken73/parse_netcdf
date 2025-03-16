import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import json
from pathlib import Path;
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# def generate_image_from_data(data, output_path, image_size=(1024, 768)):
#     """
#     주어진 [[lon, lat, value], ...] 데이터를 이미지로 변환.
    
#     Parameters:
#     - data: [[lon, lat, value], ...] 형식의 데이터 리스트
#     - output_path: 생성된 이미지를 저장할 파일 경로 (예: 'output.png')
#     - image_size: 생성할 이미지 크기 (width, height)
#     """
#     # 데이터 분리
#     lons = np.array([d[0] for d in data])
#     lats = np.array([d[1] for d in data])
#     values = np.array([d[2] for d in data])
    
#     # 경계 설정 (Mercator 투영을 고려한 위경도 범위)
#     lon_min, lon_max = lons.min(), lons.max()
#     lat_min, lat_max = lats.min(), lats.max()
    
#     # 이미지 크기에 맞게 그리드 생성
#     lon_grid = np.linspace(lon_min, lon_max, image_size[0])
#     lat_grid = np.linspace(lat_max, lat_min, image_size[1])  # 위도는 상단에서 하단으로 감소
#     lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    
#     # 데이터를 그리드에 보간
#     grid_values = griddata(
#         (lons, lats), values, (lon_grid, lat_grid), method='cubic', fill_value=np.nan
#     )
    
#     # 색상 보간 함수 (온도 -> RGB)
#     def get_color_from_value(value):
#         if np.isnan(value):
#             return [0, 0, 0, 0]  # 투명한 검정색 (RGBA)
#         t_min, t_max = -100, 30
#         ratio = min(max((value - t_min) / (t_max - t_min), 0), 1)
#         r = int(255 * (1 - ratio))
#         g = int(255 * (1 - ratio))
#         b = int(255 * (1 - ratio))
#         return [r, g, b, 255]  # RGBA
    
#     # 이미지 데이터 생성 (RGBA)
#     image_data = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)
#     for i in range(image_size[1]):
#         for j in range(image_size[0]):
#             image_data[i, j] = get_color_from_value(grid_values[i, j])
    
#     # 이미지를 PNG로 저장
#     image = Image.fromarray(image_data, 'RGBA')
#     image.save(output_path)
    
#     # 경계 좌표 반환 (Deck.gl에서 사용)
#     bounds = [lon_min, lat_min, lon_max, lat_max]
#     return bounds

# def generate_image_from_data(data, output_path, image_size=(1024, 768), bounds=None):
#     """
#     주어진 [[lon, lat, value], ...] 데이터를 이미지로 변환.
#     bounds는 [lon_min, lat_min, lon_max, lat_max] 형식.
#     """
#     # 데이터 분리
#     lons = np.array([d[0] for d in data])
#     lats = np.array([d[1] for d in data])
#     values = np.array([d[2] for d in data])
    
#     # 경계 설정 (제공된 bounds가 없으면 데이터에서 추출)
#     if bounds is None:
#         lon_min, lon_max = lons.min(), lons.max()
#         lat_min, lat_max = lats.min(), lats.max()
#     else:
#         lon_min, lat_min, lon_max, lat_max = bounds
    
#     # 이미지 크기에 맞게 그리드 생성
#     lon_grid = np.linspace(lon_min, lon_max, image_size[0])
#     lat_grid = np.linspace(lat_max, lat_min, image_size[1])  # 위도는 상단에서 하단으로
#     lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    
#     # 데이터 보간 (경계 외부는 -9999로 처리)
#     grid_values = griddata(
#         (lons, lats), values, (lon_grid, lat_grid), method='linear', fill_value=-9999
#     )
    
#     # 색상 보간 함수 (값 -> RGB, -9999는 투명 처리)
#     def get_color_from_value(value):
#         if value == -9999:
#             return [0, 0, 0, 0]  # 투명
#         t_min, t_max = -100, 30
#         ratio = min(max((value - t_min) / (t_max - t_min), 0), 1)
#         r = int(255 * (1 - ratio))
#         g = int(255 * (1 - ratio))
#         b = int(255 * (1 - ratio))
#         return [r, g, b, 255]
    
#     # 이미지 데이터 생성 (RGBA)
#     image_data = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)
#     for i in range(image_size[1]):
#         for j in range(image_size[0]):
#             image_data[i, j] = get_color_from_value(grid_values[i, j])
    
#     # 이미지를 PNG로 저장
#     image = Image.fromarray(image_data, 'RGBA')
#     image.save(output_path)
    
#     return [lon_min, lat_min, lon_max, lat_max]

## griddata 없이 생성
def generate_image_from_data(data, output_path, image_size=(600, 520), bounds=[76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447]):
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
    
    # 데이터를 2D 배열에 매핑
    for lon, lat, value in data:
        # 위경도를 픽셀 인덱스로 변환
        col = int((lon - lon_min) / lon_step)  # 경도 -> 열 인덱스
        row = int((lat_max - lat) / lat_step)  # 위도 -> 행 인덱스 (위도가 상단에서 하단으로 감소)
        
        # 인덱스가 범위 내에 있는지 확인
        if 0 <= row < height and 0 <= col < width:
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
    # data = [
    #     [124.5, 33.0, 20],
    #     [124.55, 33.0, 15],
    #     [124.5, 33.05, 10],
    #     # ... 추가 데이터
    # ]
    data = json.load(data_json)
    output_path = Path(data_file).stem + '.png'
    bounds = generate_image_from_data(data, output_path, image_size=(600, 520))
    # bounds = generate_image_from_data(data, output_path, image_size=(1024, 768))
    print(f"Image bounds: {bounds}")