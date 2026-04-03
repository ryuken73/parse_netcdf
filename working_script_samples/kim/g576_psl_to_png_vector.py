import xarray as xr
import numpy as np
from PIL import Image
import os
import glob
from concurrent.futures import ThreadPoolExecutor

# 설정
MIN_P = 900.0
MAX_P = 1100.0
INPUT_PATH = './kim_nc/*.nc'
OUTPUT_DIR = './kim_png_10min'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 이미지 저장 병렬 처리 함수
def save_image_task(img_array, output_path):
    Image.fromarray(img_array, mode='L').save(output_path)

def process_fast_interpolation():
    print("1. 파일 목록 확인 중...")
    # 파일 이름순(ft000, ft003...) 정렬 보장
    files = sorted(glob.glob(INPUT_PATH))
    if len(files) < 2:
        print("최소 2개 이상의 nc 파일이 필요합니다.")
        return

    total_files = len(files)
    # 3시간 = 180분, 10분 단위면 두 파일 사이에 18개 구간 생성
    steps_between_files = 18 
    
    print(f"2. 총 {total_files}개 파일, Pairwise 보간 및 변환 시작...")
    
    global_frame_idx = 0
    
    # ThreadPoolExecutor를 사용해 이미지 저장 속도 향상
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(total_files - 1):
            file1 = files[i]
            file2 = files[i+1]
            
            # 1. 두 개의 파일만 메모리에 로드
            with xr.open_dataset(file1) as ds1, xr.open_dataset(file2) as ds2:
                # 불필요한 차원(time 등) 제거 및 hPa 단위 변환
                val1 = np.squeeze(ds1['psl'].values) / 100.0
                val2 = np.squeeze(ds2['psl'].values) / 100.0

            # 2. Vectorized Interpolation (NumPy 브로드캐스팅)
            # 가중치 배열 생성 (0.0 ~ 0.944... 까지 18단계) 
            # endpoint=False로 해야 T1~T2 사이를 중복 없이 처리 가능
            weights = np.linspace(0, 1, steps_between_files, endpoint=False)
            
            # weights를 (18, 1, 1) 형태로 변환하여 2D 데이터(lat, lon)에 곱함
            weights = weights[:, np.newaxis, np.newaxis]
            
            # 18개의 프레임이 한 번의 연산으로 생성됨! shape: (18, lat, lon)
            interp_vals = val1 + (val2 - val1) * weights
            
            # 3. Vectorized Clipping & Normalization
            interp_clipped = np.clip(interp_vals, MIN_P, MAX_P)
            interp_norm = ((interp_clipped - MIN_P) / (MAX_P - MIN_P) * 255).astype(np.uint8)
            
            # 4. 생성된 18개 배열을 병렬로 PNG 저장
            for step_idx in range(steps_between_files):
                output_path = os.path.join(OUTPUT_DIR, f"frame_{global_frame_idx:04d}.png")
                # 스레드 풀에 작업 위임
                executor.submit(save_image_task, interp_norm[step_idx], output_path)
                global_frame_idx += 1
                
            print(f"진행률: {i+1}/{total_files-1} 파일 쌍 처리 완료 (프레임 {global_frame_idx}까지 생성)")

        # 마지막 파일(T_end)의 정확한 마지막 프레임 이미지 한 장 추가 저장
        with xr.open_dataset(files[-1]) as ds_last:
            val_last = np.squeeze(ds_last['psl'].values) / 100.0
            val_clipped = np.clip(val_last, MIN_P, MAX_P)
            val_norm = ((val_clipped - MIN_P) / (MAX_P - MIN_P) * 255).astype(np.uint8)
            
            output_path = os.path.join(OUTPUT_DIR, f"frame_{global_frame_idx:04d}.png")
            executor.submit(save_image_task, val_norm, output_path)
            global_frame_idx += 1

    print(f"완료! {OUTPUT_DIR} 폴더에 총 {global_frame_idx}장의 이미지가 초고속으로 생성되었습니다.")

if __name__ == "__main__":
    process_fast_interpolation()