import json
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom, gaussian_filter

# 1. 데이터 로드
json_file = 'gfs_0p25_wind_850mb_202603162200_202603170700.json'
with open(json_file, 'r') as f:
    gfs_data = json.load(f)

u_data = np.array(gfs_data[0]['data']).reshape(721, 1440)
v_data = np.array(gfs_data[1]['data']).reshape(721, 1440)

# 2. 풍속 계산
magnitude = np.sqrt(u_data**2 + v_data**2)
magnitude = magnitude * 3.6  # m/s -> km/h 변환

# 3. 고차원 보간 (Bicubic Interpolation)
# 721x1440 데이터를 1024x2048로 약 1.4배 정교하게 확대
# order=3은 Bicubic 보간을 의미하며 훨씬 부드러운 결과물을 만듭니다.
zoom_factor = (2048 / 721, 4096 / 1440)
high_res_data = zoom(magnitude, zoom_factor, order=3)

# 4. 가우시안 필터 적용 (방법 2의 핵심: 동영상 같은 부드러움)
# sigma가 높을수록 구름처럼 뭉실뭉실해집니다. 1.0~2.0 사이에서 조절해 보세요.
smoothed_data = gaussian_filter(high_res_data, sigma=1.5)

# 4. 목표 이미지와 유사한 커스텀 컬러맵 생성 (어두운 파랑 -> 진한 초록 -> 밝은 노랑초록)
# colors = [
#     (0.0, 0.0, 0.3),  # 매우 어두운 파랑 (저풍속)
#     (0.0, 0.2, 0.5),  # 남색
#     (0.0, 0.5, 0.2),  # 진한 초록
#     (0.5, 0.8, 0.1),  # 밝은 연두
#     (0.9, 1.0, 0.3)   # 노란빛 초록 (고풍속)
# ]
WIND_COLORS = [
    # (0.0000, (0.0000, 0.0157, 0.9961)),  # x: 231, #0004fe
    # (0.0509, (0.0000, 0.4431, 0.8941)),  # x: 253, #0071e4
    # (0.0903, (0.0000, 0.7176, 0.6902)),  # x: 270, #00b7b0
    # (0.1250, (0.0000, 0.8902, 0.4471)),  # x: 285, #00e372
    # (0.2060, (0.2314, 0.9686, 0.0000)),  # x: 320, #3bf700
    # (0.2731, (0.7255, 0.6784, 0.0000)),  # x: 349, #b9ad00
    # (0.4074, (0.9137, 0.0000, 0.4000)),  # x: 407, #e90066
    # (0.4421, (0.7490, 0.0000, 0.6549)),  # x: 422, #bf00a7
    # (0.7037, (0.8392, 0.4549, 0.8392)),  # x: 535, #d674d6
    # (1.0000, (0.9961, 0.9922, 0.9961)),  # x: 663, #fefdfe
    (0.0000, (0.01, 0.01, 0.05)),  # 0km/h: 거의 검은색에 가까운 남색
    (0.02,   (0.02, 0.05, 0.20)),  # 저풍속: 어두운 파랑
    (0.10,   (0.00, 0.40, 0.50)),  # 중간: 청록색
    (0.20,   (0.00, 0.80, 0.30)),  # 강풍: 선명한 초록
    (0.4074, (0.9137, 0.0000, 0.4000)),  # x: 407, #e90066
    (0.4421, (0.7490, 0.0000, 0.6549)),  # x: 422, #bf00a7
    (0.7037, (0.8392, 0.4549, 0.8392)),  # x: 535, #d674d6
    (1.00,   (1.00, 1.00, 1.00)),  # 최고속: 흰색    
]

custom_cmap = LinearSegmentedColormap.from_list('wind_map', WIND_COLORS, N=1024)

# 5. 고해상도 렌더링
plt.figure(figsize=(20.48, 10.24), dpi=100) # 2048 x 1024 해상도 설정

# 가우시안 필터를 살짝 적용하여 데이터 노이즈 제거 (품질 향상의 핵심)
plt.imshow(smoothed_data, cmap=custom_cmap, aspect='auto', 
           extent=[0, 360, -90, 90], interpolation='bicubic',
           vmin=0, vmax=360)  # 풍속 범위에 맞게 vmin, vmax 조절

plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# 6. 저장 (파일 용량을 위해 최적화된 저장)
save_path = 'wind_final_ultra_res.png'
plt.savefig(save_path, dpi=100, facecolor=(0,0,0.3))
plt.close()  # 메모리 해제

# [Step 3] Web Mercator 변환 프로세스
temp_tif = save_path.replace(".png", "_temp.tif")
merc_png = save_path.replace(".png", "_merc.png") # 웹용 파일명 구분

# 1. EPSG:4326 좌표 정보 주입 (TIF 생성)
subprocess.run([
    'gdal_translate', '-of', 'GTiff', 
    '-a_srs', 'EPSG:4326', 
    '-a_ullr', '0', '90', '360', '-90', 
    save_path, temp_tif
], check=True, stdout=subprocess.DEVNULL)

# 2. EPSG:3857로 투영 변환 (Mercator)
subprocess.run([
    'gdalwarp', '-overwrite', '-t_srs', 'EPSG:3857', '-s_srs', 'EPSG:4326',
    '-te', '-20037508.34', '-20037508.34', '20037508.34', '20037508.34',
    '-r', 'bilinear', '-of', 'PNG',
    temp_tif, merc_png
], check=True, stdout=subprocess.DEVNULL)

# [Step 4] 임시 파일 및 불필요한 파일 정리
if os.path.exists(temp_tif): os.remove(temp_tif)
aux_xml = merc_png + ".aux.xml"
if os.path.exists(aux_xml): os.remove(aux_xml)

print(f"[SUCCESS] Generated Equi & Mercator for WIND")

# plt.show()

print("목표 이미지 품질의 고해상도 배경 생성 완료")