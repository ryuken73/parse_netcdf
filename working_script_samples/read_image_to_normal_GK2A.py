import numpy as np
from matplotlib.image import imread, imsave
from matplotlib.pyplot import imsave as plt_imsave

def create_normal_map_from_height_map(height_file='height_map.png', output_normal_path='normal_map.png', height_scale=1.0):
    # Compute normal map from height_map (using original float heights for better precision)
    # Use np.gradient for derivatives

    # 이미지 읽기
    img = imread(height_file)
    if img.shape[2] == 4:  # 알파 채널 제거
        img = img[:, :, :3]

    # uint8로 변환
    img_uint8 = (img * 255).astype(np.uint8)

    # 벡터화된 높이 맵 생성
    # height_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    # G 채널 사용널
    height_map = img_uint8[:, :, 1].astype(np.float32)

    dy, dx = np.gradient(height_map)
    
    # Normal vector: (-dx, -dy, 1) normalized, scaled by height_scale
    normals = np.dstack((-dx * height_scale, -dy * height_scale, np.ones_like(height_map)))
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.where(norms == 0, 1, norms)  # Avoid division by zero
    
    # Map to [0,1] for image
    normals = (normals + 1) / 2
    
    # Save normal map
    plt_imsave(output_normal_path, normals)

height_file = f'E:\\cloud_images\\fd\\mono_o\\gk2a_ami_le1b_ir105_fd020ge_202509241530_202509250030_step1_mono.png'
create_normal_map_from_height_map(
    height_file=height_file,
    output_normal_path='normal_map_GK2A.png'
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
