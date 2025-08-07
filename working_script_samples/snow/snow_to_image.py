# refactor
# -*- coding:utf8 -*-
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import json
from pyproj import CRS, Transformer
import geopandas
import pandas as pd
from shapely.geometry import Point

# --- 1. 데이터 로드 및 좌표 변환 함수 ---
def load_and_transform_data(json_path: str, transformer: Transformer):
    """
    JSON 파일에서 데이터를 로드하고, 경위도 좌표를 Web Mercator로 변환합니다.
    """
    with open(json_path, encoding='UTF-8') as f:
        data = json.load(f)
    print('file load success')

    lons = np.array([float(item["lon"]) for item in data])
    lats = np.array([float(item["lat"]) for item in data])
    snow_depths = np.array([float(item["sd"]) for item in data])

    station_xs_wm, station_ys_wm = transformer.transform(lons, lats)

    return lons, lats, snow_depths, station_xs_wm, station_ys_wm

# --- 2. 보간 수행 함수 ---
def perform_interpolation(station_xs_wm, station_ys_wm, snow_depths,
                          min_x_wm, max_x_wm, min_y_wm, max_y_wm,
                          image_width_pixels, image_height_pixels, epsilon=0.1):
    """
    Web Mercator 좌표에서 RBF 보간을 수행하고, 보간된 그리드 데이터를 반환합니다.
    """
    xi_wm = np.linspace(min_x_wm, max_x_wm, image_width_pixels)
    yi_wm = np.linspace(min_y_wm, max_y_wm, image_height_pixels)
    XI_WM, YI_WM = np.meshgrid(xi_wm, yi_wm)

    rbf = Rbf(station_xs_wm, station_ys_wm, snow_depths, function='linear', epsilon=epsilon, smooth=0)
    ZI_WM = rbf(XI_WM, YI_WM)

    return XI_WM, YI_WM, ZI_WM

# --- 3. 마스킹 및 이미지 생성 함수 ---
def create_masked_image(ZI_WM, XI_WM, YI_WM, snow_depths, station_xs_wm, station_ys_wm,
                        image_width_pixels, image_height_pixels,
                        korea_geojson_path: str):
    """
    보간된 데이터에 마스킹을 적용하고, Matplotlib를 사용하여 이미지를 생성합니다.
    """
    try:
        korea_boundary = geopandas.read_file(korea_geojson_path)
        print('Korea boundary GeoJSON loaded successfully.')

        korea_boundary_wm = korea_boundary.to_crs(epsg=3857)

        grid_points_df = pd.DataFrame({
            'x': XI_WM.ravel(),
            'y': YI_WM.ravel(),
            'index': range(len(XI_WM.ravel()))
        })
        grid_points_gdf = geopandas.GeoDataFrame(
            grid_points_df,
            geometry=geopandas.points_from_xy(grid_points_df['x'], grid_points_df['y']),
            crs="EPSG:3857"
        )

        transformer = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:3857"), always_xy=True)
        lat_cutoff = 37.7
        _, y_cutoff_wm = transformer.transform(127, lat_cutoff)
        points_to_check_gdf = grid_points_gdf[grid_points_gdf['y'] >= y_cutoff_wm].copy()

        korea_boundary_union = korea_boundary_wm.geometry.unary_union
        points_within_korea = points_to_check_gdf.geometry.within(korea_boundary_union)

        mask = np.zeros_like(ZI_WM, dtype=bool)
        mask_indices = points_to_check_gdf[~points_within_korea]['index'].values
        mask.ravel()[mask_indices] = True

        # 음수 또는 아주 작은 값 마스킹 (0.05 미만)
        mask[ZI_WM < 0.05] = True

        Z_masked = np.ma.array(ZI_WM, mask=mask)

        print(f'Number of grid points checked: {len(points_to_check_gdf)} out of {len(grid_points_gdf)}.')

    except Exception as e:
        print(f"Error loading or processing GeoJSON: {e}")
        print("Proceeding without boundary masking. The output will show data outside Korea.")
        mask = np.zeros_like(ZI_WM, dtype=bool)
        mask[ZI_WM < 0.05] = True
        Z_masked = np.ma.array(ZI_WM, mask=mask)

    # --- Matplotlib 이미지 생성 및 저장 ---
    print('image_width_pixels', image_width_pixels)
    print('image_height_pixels', image_height_pixels)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()

    ax.set_aspect('equal')

    # 'cool' 컬러맵 사용
    contour = ax.contourf(XI_WM, YI_WM, Z_masked, levels=20, cmap='twilight', extend='neither', alpha=1)

    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)

    for i in range(len(lons)):
        if snow_depths[i] > 0.0:
            ax.text(station_xs_wm[i], station_ys_wm[i], f'{snow_depths[i]:.1f}',
                    ha='center', va='center', color='red', fontsize=5)

    fig.savefig('snow_depth_mercator.png', bbox_inches='tight', pad_inches=0, transparent=True, facecolor='none')
    plt.close(fig)
    print("Image 'snow_depth_mercator.png' generated.")


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 설정 변수
    JSON_FILE_PATH = 'SNOW_24H_202507051300_ALL.json'
    GEOJSON_FILE_PATH = 'skorea-provinces-2018-geo.json'

    min_lon, max_lon = 124.4, 131.6
    min_lat, max_lat = 33, 38.6
    image_width_pixels = 200
    image_height_pixels = 200
    epsilon = 0.1

    # 1. 데이터 로드 및 변환
    transformer = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:3857"), always_xy=True)
    lons, lats, snow_depths, station_xs_wm, station_ys_wm = load_and_transform_data(JSON_FILE_PATH, transformer)

    # 2. 그리드 범위 및 보간 데이터 생성
    wm_coords_top_left = transformer.transform(min_lon, max_lat)
    wm_coords_top_right = transformer.transform(max_lon, max_lat)
    wm_coords_bottom_left = transformer.transform(min_lon, min_lat)

    min_x_wm = wm_coords_top_left[0]
    max_x_wm = wm_coords_top_right[0]
    min_y_wm = wm_coords_bottom_left[1]
    max_y_wm = wm_coords_top_left[1]

    XI_WM, YI_WM, ZI_WM = perform_interpolation(
        station_xs_wm, station_ys_wm, snow_depths,
        min_x_wm, max_x_wm, min_y_wm, max_y_wm,
        image_width_pixels, image_height_pixels, epsilon
    )

    # 3. 마스킹 및 이미지 생성
    create_masked_image(
        ZI_WM, XI_WM, YI_WM, snow_depths, station_xs_wm, station_ys_wm,
        image_width_pixels, image_height_pixels,
        GEOJSON_FILE_PATH
    )