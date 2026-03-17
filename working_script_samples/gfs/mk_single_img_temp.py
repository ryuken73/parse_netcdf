import json
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom, gaussian_filter

# 1. 데이터 로드
file_path = 'gfs_0p25_tmp_500mb_202603162200_202603170700.json'
with open(file_path, 'r') as f:
    gfs_data = json.load(f)

# 데이터 추출 (721 x 1440)
raw_data = np.array(gfs_data[0]['data']).reshape(721, 1440)

# 2. 단위 변환 (K -> °C)
temp_celsius = raw_data - 273.15

# 3. 고해상도 보간 (2048 x 1024)
target_shape = (2048, 4096)
zoom_factor = (target_shape[0] / temp_celsius.shape[0], target_shape[1] / temp_celsius.shape[1])
high_res_data = zoom(temp_celsius, zoom_factor, order=3)

# 4. 가우시안 필터 적용 (방법 2의 핵심: 동영상 같은 부드러움)
# sigma가 높을수록 구름처럼 뭉실뭉실해집니다. 1.0~2.0 사이에서 조절해 보세요.
# smoothed_data = gaussian_filter(high_res_data, sigma=1.5)

# 5. 다단계 컬러맵 구성 (N=1024로 분해능 극대화)
# 보내주신 컬러바의 색상 흐름을 더 세밀하게 정의했습니다.
# colors = [
#     (0.2, 0.0, 0.4),  # -80도: 진한 보라
#     (0.0, 0.0, 0.6),  # 남색
#     (0.0, 0.3, 0.8),  # 파랑
#     (0.0, 0.7, 0.9),  # 하늘색
#     (0.0, 0.6, 0.3),  # 청록
#     (0.0, 0.8, 0.0),  # 초록
#     (0.7, 0.9, 0.0),  # 연두
#     (1.0, 1.0, 0.0),  # 노랑
#     (1.0, 0.6, 0.0),  # 주황
#     (0.9, 0.2, 0.0),  # 다홍
#     (0.8, 0.0, 0.0),  # 빨강
#     (0.4, 0.0, 0.0)   # 55도: 검붉은색
# ]
# N=1024 단계를 사용하여 색상 끊김 현상(Banding)을 방지합니다.
# custom_cmap = LinearSegmentedColormap.from_list('smooth_tmp_map', colors, N=1024)

# 1. 시트 데이터를 기반으로 완성된 컬러 위치 및 RGB 리스트
noaa_colors_pos = [
    (0.0000, (0.1451, 0.0157, 0.1725)), (0.0315, (0.1490, 0.0235, 0.2824)),
    (0.0921, (0.1608, 0.0392, 0.5059)), (0.1506, (0.2549, 0.1098, 0.2980)),
    (0.1753, (0.2941, 0.1373, 0.2078)), (0.2067, (0.3882, 0.1529, 0.2275)),
    (0.2315, (0.4902, 0.1529, 0.3255)), (0.2764, (0.6745, 0.1490, 0.5098)),
    (0.3393, (0.6196, 0.3333, 0.6549)), (0.3640, (0.5490, 0.4353, 0.6902)),
    (0.3910, (0.4706, 0.5490, 0.7333)), (0.4135, (0.4078, 0.6431, 0.7686)),
    (0.4472, (0.3098, 0.7882, 0.8196)), (0.4809, (0.2431, 0.7608, 0.8235)),
    (0.5506, (0.1412, 0.4863, 0.7647)), (0.5933, (0.0824, 0.3451, 0.6667)),
    # (0.6112, (0.1294, 0.5333, 0.0627)), (0.6472, (0.3961, 0.6784, 0.1137)),
    # (0.7146, (0.8941, 0.9451, 0.2157)), (0.7416, (0.9529, 0.8745, 0.1804)),
    # (0.7685, (0.9294, 0.7020, 0.1020)), (0.8022, (0.9176, 0.5569, 0.1020)),
    # (0.8449, (0.9059, 0.3843, 0.1333)), (0.9101, (0.7373, 0.2235, 0.1843)),
    (0.6112, (0.1294, 0.5333, 0.0627)), # 녹색
    (0.6472, (0.3961, 0.6784, 0.1137)), # 연두
    (0.7000, (0.80, 0.85, 0.20)),       # 노란색 지점을 0.71에서 0.70으로 당기고 채도 낮춤
    (0.7300, (0.85, 0.60, 0.10)),       # 주황색 진입을 더 빨리 하여 노란색 영역 축소
    (0.7800, (0.80, 0.40, 0.10)),       # 깊은 주황 (태평양 메인 톤)
    (0.8449, (0.85, 0.25, 0.10)),       # 다홍색
    (0.9281, (0.6549, 0.2000, 0.2000)), (0.9596, (0.5216, 0.1608, 0.2275)),
    (0.9708, (0.4706, 0.1451, 0.2353)), (0.9955, (0.3608, 0.1098, 0.2588)),
    (1.0000, (0.3451, 0.1059, 0.2627))
]

# 2. 전역적으로 사용할 컬러맵 객체 생성
noaa_cmap = LinearSegmentedColormap.from_list('noaa_temp_map', noaa_colors_pos, N=1024)

# 6. 렌더링
plt.figure(figsize=(20.48, 10.24), dpi=100)

plt.imshow(high_res_data, cmap=noaa_cmap, aspect='auto', 
           extent=[0, 360, -90, 90], interpolation='bicubic',
           vmin=-80, vmax=55)

plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# 7. 저장
save_name = 'gfs_temp_smooth_no_gaussian.png'
plt.savefig(save_name, dpi=100)
# plt.show()
plt.close()

# [Step 3] Web Mercator 변환 프로세스
temp_tif = save_name.replace(".png", "_temp.tif")
merc_png = save_name.replace(".png", "_merc.png") # 웹용 파일명 구분

# 1. EPSG:4326 좌표 정보 주입 (TIF 생성)
subprocess.run([
    'gdal_translate', '-of', 'GTiff', 
    '-a_srs', 'EPSG:4326', 
    '-a_ullr', '0', '90', '360', '-90', 
    save_name, temp_tif
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


print(f"방법 2(부드러운 전이) 적용 이미지 생성 완료: {save_name}")