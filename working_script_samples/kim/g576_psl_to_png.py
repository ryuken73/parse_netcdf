import xarray as xr
import numpy as np
from PIL import Image
import os

# 설정
MIN_P = 900.0
MAX_P = 1100.0
INPUT_PATH = './kim_nc/*.nc'
OUTPUT_DIR = './kim_png_10min'
DROP_VARS = ['time_ini_bnds', 'time_bnds', 'soil_levs_bnds']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_10min_interpolation():
    print("1. 데이터 로드 및 10분 단위 보간 시작...")
    
    # 8개 파일을 합치고 10분 단위로 선형 보간
    ds = xr.open_mfdataset(INPUT_PATH, combine='nested', concat_dim='time', drop_variables=DROP_VARS)
    ds = ds.drop_duplicates('time').sortby('time')
    
    # '10min' 리샘플링 후 선형 보간
    ds_interp = ds.resample(time='10min').interpolate('linear')
    
    total_frames = len(ds_interp.time)
    print(f"2. 총 {total_frames}개의 프레임 생성 중...")

    for i in range(total_frames):
        data_step = ds_interp.isel(time=i)
        psl_val = data_step['psl'].values
        
        if psl_val.ndim > 2: psl_val = psl_val[0]
            
        psl_hpa = psl_val / 100.0
        psl_clipped = np.clip(psl_hpa, MIN_P, MAX_P)
        psl_norm = ((psl_clipped - MIN_P) / (MAX_P - MIN_P) * 255).astype(np.uint8)
        
        output_path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png")
        Image.fromarray(psl_norm, mode='L').save(output_path)
        
        if i % 20 == 0:
            print(f"진행률: {i}/{total_frames} ({int(i/total_frames*100)}%)")

    print(f"완료! {OUTPUT_DIR} 폴더에 {total_frames}장의 이미지가 생성되었습니다.")

if __name__ == "__main__":
    process_10min_interpolation()