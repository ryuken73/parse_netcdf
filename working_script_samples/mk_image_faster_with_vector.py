# vector 연산을 기반으로 이미지를 빠르게 만드는 모듈
# 보간은 없음 (보간 로직 만들기 어려움)

import numpy as np
from PIL import Image, ImageFilter
from pyproj import Transformer, CRS

# Web Mercator 투영 정의
web_mercator_crs = CRS.from_epsg("3857")
wgs84_crs = CRS.from_epsg("4326")
transformer_to_mercator = Transformer.from_crs(wgs84_crs, web_mercator_crs, always_xy=True)

# get_color_from_temperature 함수 (기존과 동일)
def get_color_from_temperature(temp):
    colors = {
        20: [255,255,255], -20: [255, 255, 255], -21: [135, 206, 235], -30: [0, 0, 255],
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

# def get_mono_color_from_value(value):
#     if value == -9999:
#         return [0, 0, 0, 0]  # 투명
#         # print(f'fill red {value}')
#         return [255, 0, 0, 255]  # red
#     t_min, t_max = -100, 30
#     ratio = min(max((value - t_min) / (t_max - t_min), 0), 1)
#     r = int(255 * (1 - ratio))
#     g = int(255 * (1 - ratio))
#     b = int(255 * (1 - ratio))
#     return [r, g, b, 255]  # RGBA

def get_mono_color_from_value(value, factor=3):
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

def generate_image_from_data_fast(data, output_path, image_size=(600, 520), bounds=[60, -80, 180, 80]):
    """
    NumPy 배열 [[lon, lat, value], ...] 데이터를 Web Mercator 투영으로 이미지 변환.
    """
    # 경계 설정
    lon_min, lat_min, lon_max, lat_max = bounds
    width, height = image_size

    # WGS84 -> Web Mercator 변환
    x_min, y_max = transformer_to_mercator.transform(lon_min, lat_max)
    x_max, y_min = transformer_to_mercator.transform(lon_max, lat_min)
    if x_max < 0:
        x_max = 2 * 20037508.342789244 + x_max  # 음수 보정
    print(f"x_min, x_max", x_min, x_max)
    print(f"y_min, y_max", y_min, y_max)

    # 픽셀 간격 계산
    x_step = (x_max - x_min) / (width - 1)
    y_step = (y_max - y_min) / (height - 1)
    print(f"x_step", x_step)
    print(f"y_step", y_step)

    # 데이터 분리
    lons = data[:, 0]
    lats = data[:, 1]
    values = data[:, 2]

    # Web Mercator로 변환
    x, y = transformer_to_mercator.transform(lons, lats)
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
        flat_colors = np.array([get_color_from_temperature(val) if val != -9999 else [0, 0, 0, 0] 
                                for val in flat_values])
        # flat_colors = np.array([get_mono_color_from_value(val) if val != -9999 else [0, 0, 0, 0] 
        #                         for val in flat_values])
        colors = flat_colors.reshape(height, width, 4)
        return colors

    # 이미지 데이터 생성
    image_data = apply_color_mapping(grid_values)
    print('length of image_dat:', len(image_data))
    print('sample of image_dat:', image_data[:100])
    print('sample of image_dat:', image_data[100:200])
    print('shape and size of image_dat:', image_data.shape, image_data.size, image_data.dtype)

    # PNG 저장
    image = Image.fromarray(image_data.astype(np.uint8), 'RGBA')
    image.save(output_path)

    # image_with_shadow = apply_lighting_and_emboss_effect(image)
    # image_with_shadow.save(output_path+'_shadow.png')
    print(f"Image saved to {output_path} with bounds: {bounds}")
    return bounds

# 사용 예시
# output_path = "output_image.png"
# bounds = [60, -80, 180, 80]
# generate_image_from_data(result, output_path, image_size=(600, 520), bounds=bounds)