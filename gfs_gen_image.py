import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom, gaussian_filter
from glob import glob

# =================================================================
# 시각화 설정 (컬러맵 및 범위)
# =================================================================

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

VIS_CONFIG = {
    "TMP": {
        "cmap": LinearSegmentedColormap.from_list('tmp', NOAA_TMP_COLORS, N=1024),
        "vmin": -80, "vmax": 55, "sigma": 0, "convert_celsius": True, "facecolor": 'black'
    },
    "RH": {
        "cmap": LinearSegmentedColormap.from_list('rh', RH_COLORS, N=1024),
        "vmin": 0, "vmax": 100, "sigma": 0, "convert_celsius": False, "facecolor": 'black'
    },
    "WIND": {
        "cmap": LinearSegmentedColormap.from_list('wind', WIND_COLORS, N=1024),
        "vmin": 0, "vmax": 50, "sigma": 1.5, "convert_celsius": False, "facecolor": (0, 0, 0.3)
    }
}

TARGET_SHAPE = (1024, 2048)

# =================================================================
# 공용 시각화 로직 (_com)
# =================================================================

def process_and_save_image_com(data_list, type_key, save_path):
    """
    JSON 데이터 리스트를 받아 PNG 이미지를 생성 및 저장하는 핵심 함수
    """
    cfg = VIS_CONFIG.get(type_key.upper())
    if not cfg:
        print(f"      [WARN] No VIS_CONFIG for {type_key}")
        return

    try:
        # 1. 데이터 배열화 (Wind는 Magnitude 계산)
        if type_key.upper() == "WIND" and len(data_list) >= 2:
            u = np.array(data_list[0]['data']).reshape(721, 1440)
            v = np.array(data_list[1]['data']).reshape(721, 1440)
            mag = np.sqrt(u**2 + v**2)
        else:
            mag = np.array(data_list[0]['data']).reshape(721, 1440)

        # 2. 단위 변환 (K -> C)
        if cfg["convert_celsius"]:
            mag = mag - 273.15

        # 3. 보간 (Bicubic)
        zoom_factor = (TARGET_SHAPE[0] / mag.shape[0], TARGET_SHAPE[1] / mag.shape[1])
        high_res = zoom(mag, zoom_factor, order=3)

        # 4. 가우시안 필터
        if cfg["sigma"] > 0:
            high_res = gaussian_filter(high_res, sigma=cfg["sigma"])

        # 5. 렌더링
        plt.figure(figsize=(20.48, 10.24), dpi=100)
        plt.imshow(high_res, cmap=cfg["cmap"], aspect='auto', 
                   extent=[0, 360, -90, 90], interpolation='bicubic',
                   vmin=cfg["vmin"], vmax=cfg["vmax"])
        
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_path, dpi=100, facecolor=cfg["facecolor"])
        plt.close()
        return True
    except Exception as e:
        print(f"      [ERROR] Render failed: {e}")
        return False

# =================================================================
# 단독 실행 시 일괄 처리 로직 (Batch Mode)
# =================================================================

if __name__ == "__main__":
    from config import get_config
    config = get_config()
    # target_root = config.OUT_PATH_WIND
    target_root = 'D:\\002.Code\\002.node\\weather_api\\data\\weather\\gfs\\2026-03-17'

    
    json_files = glob(os.path.join(target_root, "**", "gfs_*.json"), recursive=True)
    print(f"Starting batch generation for {len(json_files)} files...")

    for f_path in json_files:
        f_name = os.path.basename(f_path)
        t_key = f_name.split('_')[1].upper() # gfs_tmp_... -> TMP
        p_path = f_path.replace('.json', '.png')

        if os.path.exists(p_path): continue

        with open(f_path, 'r') as f:
            d_list = json.load(f)
        
        if process_and_save_image_com(d_list, t_key, p_path):
            print(f"   [DONE] Generated: {os.path.basename(p_path)}")