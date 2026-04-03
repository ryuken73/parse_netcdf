import netCDF4 as nc
import numpy as np
from PIL import Image

def save_psl_to_png(nc_path, output_path):
    # 1. NC 파일 열기
    ds = nc.Dataset(nc_path)
    
    # 2. PSL(해면기압) 데이터 추출 및 Scale 적용
    # 실제값 = (저장값 * scale) + offset 적용된 상태로 가져옴 (netCDF4 라이브러리 자동 처리)
    psl_raw = ds.variables['psl'][0, :, :] # (time=1, lats=598, lons=1158) -> 2D
    
    # Pa 단위를 hPa로 변환 (시각화 및 범위 설정을 위해)
    psl_hpa = psl_raw / 100.0
    
    # 3. 데이터 정규화 (Min-Max Scaling)
    # 웹 브라우저와 약속할 기압 범위 (예: 900hPa ~ 1100hPa)
    MIN_P = 900.0
    MAX_P = 1100.0
    
    # 범위를 벗어나는 값 클리핑
    psl_clipped = np.clip(psl_hpa, MIN_P, MAX_P)
    
    # 0~255 범위로 변환 (8비트 PNG 한 채널 기준)
    # 더 정밀하게 하려면 (psl - MIN_P) * 65535 / (MAX_P - MIN_P) 후 16비트로 저장 가능
    psl_norm = ((psl_clipped - MIN_P) / (MAX_P - MIN_P) * 255).astype(np.uint8)
    
    # 4. 이미지 생성 (L 모드: 8-bit pixels, black and white)
    # RGB로 저장해서 다른 데이터(온도 등)를 G, B 채널에 같이 넣을 수도 있음
    img = Image.fromarray(psl_norm, mode='L')
    
    # 5. 저장 (무손실 PNG로 저장해야 수치 정보가 왜곡되지 않음)
    img.save(output_path)
    print(f"PNG 저장 완료: {output_path}")

# 사용 예시
save_psl_to_png("g576_v091_easia_etc.2byte.ft012.2026032912.nc", "psl_easia.png")