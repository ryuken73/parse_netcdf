import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom, gaussian_filter
from glob import glob

# =================================================================
# 1. 시각화 설정 (추가/수정은 여기서만 하세요)
# =================================================================

# [컬러맵 정의]
NOAA_TMP_COLORS = [
    (0.0000, (0.1451, 0.0157, 0.1725)), (0.0315, (0.1490, 0.0235, 0.2824)),
    (0.0921, (0.1608, 0.0392, 0.5059)), (0.1506, (0.2549, 0.1098, 0.2980)),
    (0.1753, (0.2941, 0.1373, 0.2078)), (0.2067, (0.3882, 0.1529, 0.2275)),
    (0.2315, (0.4902, 0.1529, 0.3255)), (0.2764, (0.6745, 0.1490, 0.5098)),
    (0.3393, (0.6196, 0.3333, 0.6549)), (0.3640, (0.5490, 0.4353, 0.6902)),
    (0.3910, (0.4706, 0.5490, 0.7333)), (0.4135, (0.4078, 0.6431, 0.7686)),
    (0.4472, (0.3098, 0.7882, 0.8196)), (0.4809, (0.2431, 0.7608, 0.8235)),
    (0.5506, (0.1412, 0.4863, 0.7647)), (0.5933, (0.0824, 0.3451, 0.6667)),
    (0.6112, (0.1294, 0.5333, 0.0627)), (0.6472, (0.3961, 0.6784, 0.1137)),
    (0.7146, (0.8941, 0.9451, 0.2157)), (0.7416, (0.9529, 0.8745, 0.1804)),
    (0.7685, (0.9294, 0.7020, 0.1020)), (0.8022, (0.9176, 0.5569, 0.1020)),
    (0.8449, (0.9059, 0.3843, 0.1333)), (0.9101, (0.7373, 0.2235, 0.1843)),
    (0.9281, (0.6549, 0.2000, 0.2000)), (0.9596, (0.5216, 0.1608, 0.2275)),
    (0.9708, (0.4706, 0.1451, 0.2353)), (0.9955, (0.3608, 0.1098, 0.2588)),
    (1.0000, (0.3451, 0.1059, 0.2627))
]
RH_COLORS = [
    (0.00, (0.35, 0.20, 0.05)), (0.20, (0.70, 0.50, 0.20)), (0.40, (0.95, 0.90, 0.70)),
    (0.60, (0.50, 0.85, 0.50)), (0.80, (0.10, 0.60, 0.30)), (0.90, (0.10, 0.40, 0.70)),
    (1.00, (0.05, 0.10, 0.40))
]
WIND_COLORS = [
    (0.0, (0.0, 0.0, 0.3)), (0.2, (0.0, 0.2, 0.5)), (0.5, (0.0, 0.5, 0.2)),
    (0.8, (0.5, 0.8, 0.1)), (1.0, (0.9, 1.0, 0.3))
]

CMAPS = {
    "TMP": LinearSegmentedColormap.from_list('tmp_map', NOAA_TMP_COLORS, N=1024),
    "RH": LinearSegmentedColormap.from_list('rh_map', RH_COLORS, N=1024),
    "WIND": LinearSegmentedColormap.from_list('wind_map', WIND_COLORS, N=1024)
}

# [기상 타입별 렌더링 설정]
VIS_CONFIG = {
    "tmp": {
        "cmap": CMAPS["TMP"],
        "vmin": -80, "vmax": 55,
        "convert_celsius": True,
        "sigma": 0,  # 가우시안 미적용 시 0
        "facecolor": None
    },
    "rh": {
        "cmap": CMAPS["RH"],
        "vmin": 0, "vmax": 100,
        "convert_celsius": False,
        "sigma": 0,
        "facecolor": None
    },
    "wind": {
        "cmap": CMAPS["WIND"],
        "vmin": 0, "vmax": 50, # 적절한 최대 풍속 설정
        "convert_celsius": False,
        # "sigma": 1.5,
        "sigma": 0,
        "facecolor": (0, 0, 0.3)
    }
}

# 보간 해상도 설정
TARGET_SHAPE = (1024, 2048)

# =================================================================
# 2. 공용 시각화 함수 (_com)
# =================================================================

def process_array_com(json_data, type_key):
    """JSON 데이터를 시각화용 numpy 배열로 변환 및 전처리"""
    # Wind의 경우 U, V 두 성분이 들어있음
    if type_key == "wind" and len(json_data) >= 2:
        u = np.array(json_data[0]['data']).reshape(json_data[0]['header']['ny'], json_data[0]['header']['nx'])
        v = np.array(json_data[1]['data']).reshape(json_data[1]['header']['ny'], json_data[1]['header']['nx'])
        data = np.sqrt(u**2 + v**2)
    else:
        data = np.array(json_data[0]['data']).reshape(json_data[0]['header']['ny'], json_data[0]['header']['nx'])
    
    cfg = VIS_CONFIG[type_key]
    
    # 단위 변환 (K -> C)
    if cfg["convert_celsius"]:
        data = data - 273.15
        
    # 고해상도 보간
    zoom_factor = (TARGET_SHAPE[0] / data.shape[0], TARGET_SHAPE[1] / data.shape[1])
    high_res = zoom(data, zoom_factor, order=3)
    
    # 가우시안 필터
    if cfg["sigma"] > 0:
        high_res = gaussian_filter(high_res, sigma=cfg["sigma"])
        
    return high_res

def save_image_com(data, type_key, save_path):
    """최종 배열을 이미지로 렌더링 후 저장"""
    cfg = VIS_CONFIG[type_key]
    
    plt.figure(figsize=(20.48, 10.24), dpi=100)
    plt.imshow(data, cmap=cfg["cmap"], aspect='auto', 
               extent=[0, 360, -90, 90], interpolation='bicubic',
               vmin=cfg["vmin"], vmax=cfg["vmax"])
    
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 배경색 설정 (Wind의 경우 facecolor 사용 가능)
    fcolor = cfg["facecolor"] if cfg["facecolor"] else 'black'
    
    plt.savefig(save_path, dpi=100, facecolor=fcolor)
    plt.close()

# =================================================================
# 3. 메인 실행부 (Batch Processing)
# =================================================================

def run_generator(target_dir):
    """지정된 디렉토리 내의 모든 JSON을 검색하여 PNG로 변환"""
    # 하위 폴더까지 모든 gfs_*.json 검색
    json_files = glob(os.path.join(target_dir, "**", "gfs_*.json"), recursive=True)
    
    print(f"Found {len(json_files)} JSON files. Starting image generation...")

    for file_path in json_files:
        try:
            # 파일명에서 타입 추출 (예: gfs_tmp_... -> tmp)
            file_name = os.path.basename(file_path)
            type_key = file_name.split('_')[1].lower() # tmp, rh, wind
            
            if type_key not in VIS_CONFIG:
                continue

            save_path = file_path.replace('.json', '.png')
            
            # 이미 파일이 존재하면 스킵 (필요시 주석 처리)
            if os.path.exists(save_path):
                # print(f"   [SKIP] {os.path.basename(save_path)}")
                continue

            with open(file_path, 'r') as f:
                json_data = json.load(f)

            # 전처리 및 렌더링
            processed_data = process_array_com(json_data, type_key)
            save_image_com(processed_data, type_key, save_path)
            
            print(f"   [DONE] {os.path.basename(save_path)}")

        except Exception as e:
            print(f"   [ERROR] Failed to process {file_path}: {e}")

if __name__ == "__main__":
    # JSON 파일이 들어있는 루트 경로를 지정하세요.
    from config import get_config
    config = get_config()
    # target_root = config.OUT_PATH_WIND
    target_root = 'D:\\002.Code\\002.node\\weather_api\\data\\weather\\gfs\\2026-03-17'
    
    run_generator(target_root)