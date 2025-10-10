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
    color_map = np.array([
        [0, 200, 255, 0], [0, 155, 245, 255], [0, 74, 245, 255], [0, 255, 0, 255], 
        [0, 190, 0, 255], [0, 140, 0, 255], [0, 90, 0, 255], [255, 255, 0, 255], 
        [255, 220, 31, 255], [249, 205, 0, 255], [224, 185, 0, 255], [204, 170, 0, 255], 
        [255, 102, 0, 255], [255, 50, 0, 255], [210, 0, 0, 255], [180, 0, 0, 255], 
        [224, 169, 255, 255], [201, 105, 255, 255], [179, 41, 255, 255], [147, 0, 228, 255], 
        [179, 180, 222, 255], [76, 78, 177, 255], [0, 3, 144, 255], [51, 51, 51, 255]
    ], dtype=np.uint8)
    color_map = color_map[:, :3]  # Drop alpha for mapping
    # intensity_map = np.array([
    #     0, 0.1, 0.5, 1,  # 하늘색
    #     2, 3, 4, 5,      # 초록색
    #     6, 7, 8, 9, 10,  # 노랑색
    #     15, 20, 25, 30,  # 빨간색
    #     40, 50, 60, 70,  # 보라색
    #     90, 110, 150     # 파란색
    # ], dtype=np.float32)
    intensity_map = np.array([
        0, 1, 2, 3,  # 하늘색
        4, 5, 6, 7,      # 초록색
        8, 9, 10, 11, 12,  # 노랑색
        15, 20, 25, 30,  # 빨간색
        40, 50, 60, 70,  # 보라색
        90, 110, 150     # 파란색
    ], dtype=np.float32)

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


in_file = 'e:\\RDR_CMP_HSP_PUB_202509132340_step1.png'

create_height_and_normal_map(
    in_file, 
    output_height_path='height_map.png', 
    output_normal_path='normal_map.png'
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
