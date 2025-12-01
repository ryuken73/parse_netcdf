import numpy as np
from matplotlib.image import imread, imsave
from matplotlib.pyplot import imsave as plt_imsave

def create_height_and_normal_map(input_path, output_height_path='height_map.png', output_normal_path='normal_map.png', color_map=None, intensity_map=None, height_scale=1.0):
    """
    Converts a color PNG image to a height map and normal map based on a color-to-height table.
    
    Args:
    - input_path: Path to the input color PNG image.
    - output_height_path: Path to save the height map (grayscale PNG).
    - output_normal_path: Path to save the normal map (RGB PNG).
    - color_map: Dictionary mapping color names or RGB tuples to height values.
    - intensity_map: List or array of height values corresponding to the colors in color_map.
    - height_scale: Scale factor for the normals (controls bump strength).
    
    Assumes the input image uses exact colors matching the table. Unmatched colors default to height 0.
    """
    
    # 주어진 color_map과 intensity_map

    # 이미지 읽기
    img = imread(input_path)
    if img.shape[2] == 4:  # 알파 채널 제거
        img = img[:, :, :3]

    # uint8로 변환
    img_uint8 = (img * 255).astype(np.uint8)

    # 벡터화된 높이 맵 생성
    height_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    # 모든 픽셀의 RGB 값을 1D 배열로 변환 (height * width, 3)
    pixels = img_uint8.reshape(-1, 3)

    # RGB 값을 인덱스로 변환하여 빠르게 매핑
    # np.searchsorted를 사용하여 color_map에 매핑
    pixel_indices = np.zeros(pixels.shape[0], dtype=np.int32)
    for i, rgb in enumerate(color_map):
        mask = np.all(pixels == rgb, axis=1)
        pixel_indices[mask] = i

    # 높이 값 매핑
    height_map_flat = intensity_map[pixel_indices]
    height_map = height_map_flat.reshape(img.shape[0], img.shape[1])

    # 높이 맵 정규화 [0, 255]로 변환
    min_h = np.min(height_map)
    max_h = np.max(height_map)
    if max_h > min_h:
        height_map_norm = ((height_map - min_h) / (max_h - min_h) * 255).astype(np.uint8)
    else:
        height_map_norm = np.zeros_like(height_map, dtype=np.uint8)

    # 높이 맵 저장
    plt_imsave(output_height_path, height_map_norm, cmap='gray')
    
    # Compute normal map from height_map (using original float heights for better precision)
    # Use np.gradient for derivatives
    dy, dx = np.gradient(height_map)
    
    # Normal vector: (-dx, -dy, 1) normalized, scaled by height_scale
    normals = np.dstack((-dx * height_scale, -dy * height_scale, np.ones_like(height_map)))
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.where(norms == 0, 1, norms)  # Avoid division by zero
    
    # Map to [0,1] for image
    normals = (normals + 1) / 2
    
    # Save normal map
    plt_imsave(output_normal_path, normals)

def read_image(file_path):
    """이미지 파일을 읽어 numpy 배열로 반환."""
    try:
        image_data = imread(file_path)
        print(f"Image read successfully from {file_path}")
        return image_data
    except Exception as e:
        print(f"Error reading image from {file_path}: {e}")
        return None

def hex_to_rgb(hex_color: str) -> tuple:
    """
    Converts a hexadecimal color string to an RGB tuple.

    Args:
        hex_color: A string representing the hexadecimal color,
                   e.g., "#RRGGBB" or "RRGGBB".

    Returns:
        A tuple of three integers (R, G, B), where each value is
        between 0 and 255.
    """
    hex_color = hex_color.lstrip("#")  # Remove '#' if present
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


#in_file = 'e:\\RDR_CMP_HSP_PUB_202509132340_step1.png'
# in_file = 'e:\\AWS_MIN_202509132330_RN_24HR_step1.png'
in_file = 'e:\\AWS_MIN_202509132330_RN_24HR_step1_keep_size_equi.png'
aws_color_list = [
    '#ffea6e', # 0 - 2
    '#ffdc1f', # 2 - 4
    '#f9cd00', # 4 - 6
    '#e0b900', # 6 - 8
    '#ccaa00', # 8 - 10
    '#69fc69', # 10 - 12
    '#1ef31e', # 12 - 14
    '#00d500', # 14 - 16
    '#00a400', # 16 - 18
    '#008000', # 18 -20 
    '#87d9ff', # 20 - 30
    '#3ec1ff', # 30 - 40
    '#07abff', # 40 - 50
    '#008dde', # 50 - 60 
    '#0077b3', # 60 - 70
    '#b3b4de', # 70 - 80
    '#8081c7', # 80 - 90
    '#4c4eb1', # 90 - 100
    '#000390', # 100 - 110
    '#da87ff', # 110 - 120
    '#c23eff', # 120 - 130
    '#ad07ff', # 130 - 140
    '#9200e4', # 140 - 160
    '#7f00bf', # 160 - 180
    '#fa8585', # 180 - 200
    '#f63e3e', # 200 - 300
    '#ee0b0b', # 300 - 400
    '#d50000', # 400 - 500
    '#bf0000', # 500 - 700
]
color_map = np.array([hex_to_rgb(c) for c in aws_color_list], dtype=np.uint8)
color_map = color_map[:, :3]  # Drop alpha for mapping
# intensity_map = np.array([
#     0, 0.1, 0.5, 1,  # 하늘색
#     2, 3, 4, 5,      # 초록색
#     6, 7, 8, 9, 10,  # 노랑색
#     15, 20, 25, 30,  # 빨간색
#     40, 50, 60, 70,  # 보라색
#     90, 110, 150     # 파란색
# ], dtype=np.float32)
# intensity_map = np.array([
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
#     10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#     20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
# ], dtype=np.float32)
intensity_map = np.array([
    0, 2, 4, 6, 
    10, 12, 14, 16,
    20, 22, 24, 26, 28, 
    30, 32, 34, 36, 
    40, 42, 44, 46,
    50, 52, 54, 56,
    60, 62, 64, 66,
], dtype=np.float32)

create_height_and_normal_map(
    in_file, 
    output_height_path='height_map.png', 
    output_normal_path='normal_map.png',
    color_map=color_map,
    intensity_map=intensity_map
) 

# img = read_image(f'e:\\RDR_CMP_HSP_PUB_202509132340_step1.png')
# if img is not None:
#     print(f"Image shape: {img.shape}, dtype: {img.dtype}")
# height, width = img.shape[0], img.shape[1]
# img_unit8 = (img * 255).astype(np.uint8)

# for i in range(height):
#     for j in range(width):
#         if img[i,j][3] != 0 : # alpha 채널이 0이 아닌 경우
#             print(f"Pixel ({i},{j}): {img_unit8[i,j]}")
