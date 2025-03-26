import numpy as np
from PIL import Image
from pyproj import Transformer, CRS

# Web Mercator 투영 정의
web_mercator_crs = CRS.from_epsg("3857")
wgs84_crs = CRS.from_epsg("4326")
transformer_to_mercator = Transformer.from_crs(wgs84_crs, web_mercator_crs, always_xy=True)

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
    return [r, g, b, 255]

def is_point_off_range(grid_values):
    """
    grid_values에서 각 픽셀이 주변 8방향 모두 -9999인지 확인하는 마스크 반환
    """
    height, width = grid_values.shape
    off_range_mask = np.full((height, width), False, dtype=bool)

    # 주변 8방향 값 추출 (경계 처리 포함)
    left = np.roll(grid_values, 1, axis=1)
    left[:, 0] = grid_values[:, 0]  # 왼쪽 경계 보정
    right = np.roll(grid_values, -1, axis=1)
    right[:, -1] = grid_values[:, -1]  # 오른쪽 경계 보정
    up = np.roll(grid_values, 1, axis=0)
    up[0, :] = grid_values[0, :]  # 위쪽 경계 보정
    down = np.roll(grid_values, -1, axis=0)
    down[-1, :] = grid_values[-1, :]  # 아래쪽 경계 보정
    up_left = np.roll(up, 1, axis=1)
    up_left[:, 0] = up[:, 0]
    up_right = np.roll(up, -1, axis=1)
    up_right[:, -1] = up[:, -1]
    down_left = np.roll(down, 1, axis=1)
    down_left[:, 0] = down[:, 0]
    down_right = np.roll(down, -1, axis=1)
    down_right[:, -1] = down[:, -1]

    # 모든 주변 픽셀이 -9999인지 확인
    off_range_mask = (
        (left == -9999) & (right == -9999) & (up == -9999) & (down == -9999) &
        (up_left == -9999) & (up_right == -9999) & (down_left == -9999) & (down_right == -9999)
    )
    return off_range_mask

def generate_image_from_data_fast(data, output_path, image_size=(600, 520), bounds=[60, -80, 180, 80]):
    """
    NumPy 배열 [[lon, lat, value], ...] 데이터를 Web Mercator 투영으로 이미지 변환.
    is_point_off_range 포함.
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

    print('grid_values sample:', grid_values[600:800])

    # is_point_off_range 적용
    # off_range_mask = is_point_off_range(grid_values)

    # 색상 매핑을 위한 벡터화 함수
    # def apply_color_mapping(values, off_mask):
    def apply_color_mapping(values):
        colors = np.zeros((height, width, 4), dtype=np.uint8)
        flat_values = values.flatten()
        flat_colors = np.array([get_color_from_temperature(val) if val != -9999 else [0, 0, 0, 0] 
                                for val in flat_values])
        # x = np.zeros((1200, 1200, 4), dtype=np.uint8)
        # x[:] = [0, 0, 255, 255] 
        # colors = x.reshape(height, width, 4)
        colors = flat_colors.reshape(height, width, 4)
        # off_range인 경우 투명 처리
        # colors[off_mask] = [0, 0, 0, 0]
        return colors

    # 이미지 데이터 생성
    # image_data = apply_color_mapping(grid_values, off_range_mask)
    image_data = apply_color_mapping(grid_values)
    print('length of image_dat:', len(image_data))
    print('sample of image_dat:', image_data[:100])
    print('sample of image_dat:', image_data[600:800])
    print('shape and size of image_dat:', image_data.shape, image_data.size, image_data.dtype)

    # PNG 저장
    image = Image.fromarray(image_data.astype(np.uint8), 'RGBA')
    image.save(output_path)
    # image.show()
    print(f"Image saved to {output_path} with bounds: {bounds}")
    return bounds

# 사용 예시
# output_path = "output_image.png"
# bounds = [60, -80, 180, 80]
# generate_image_from_data(result, output_path, image_size=(600, 520), bounds=bounds)