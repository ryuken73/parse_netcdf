# import netCDF4 as nc
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio.v2 as imageio  # PNG 저장용

# # NetCDF 파일 열기
# file_path = "ir105_fd_ge_202502170000.nc"
# ds = nc.Dataset(file_path)

# # image_pixel_values 데이터 가져오기
# image_data = ds.variables["image_pixel_values"][:]

# # 최소-최대 값 설정
# min_val, max_val = 1448, 32768

# # 정규화 (0~255 범위로 변환)
# normalized_data = np.clip((image_data - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

# # PNG 저장
# output_path = "output.png"
# imageio.imwrite(output_path, normalized_data)

# print(f"변환된 이미지 저장 완료: {output_path}")


# import netCDF4 as nc
# import numpy as np
# import imageio.v2 as imageio

# # NetCDF 파일 열기
# file_path = "ir105_fd_ge_202502170000.nc"
# ds = nc.Dataset(file_path)

# # image_pixel_values 데이터 가져오기
# image_data = ds.variables["image_pixel_values"][:].astype(np.float32)

# # 하위 5%와 상위 95% 값 계산 (대비 강조)
# low, high = np.percentile(image_data, [5, 95])

# # 히스토그램 스트레칭 적용
# stretched_data = np.clip((image_data - low) / (high - low) * 255, 0, 255).astype(np.uint8)

# # PNG 저장
# output_path = "contrast_enhanced.png"
# imageio.imwrite(output_path, stretched_data)

# print(f"대비가 강조된 이미지 저장 완료: {output_path}")


# color enhancer
import cv2
import numpy as np

def enhance_contrast(image_path, output_path, alpha=2.0, beta=0):
    """
    이미지 대비를 조정하는 함수
    :param image_path: 원본 이미지 경로
    :param output_path: 대비 조정된 이미지 저장 경로
    :param alpha: 대비 조절 계수 (기본값 2.0, 높을수록 대비 증가)
    :param beta: 밝기 조절 값 (기본값 0, 필요 시 조정)
    """
    # 이미지 읽기 (그레이스케일 변환)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    
    # 대비 조정
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 결과 저장
    cv2.imwrite(output_path, adjusted)
    print(f"대비 조정 완료: {output_path}")

# 사용 예시
enhance_contrast("input.png", "enhanced_output.png", alpha=2.0, beta=0)