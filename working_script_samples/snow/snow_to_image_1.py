# refactor
# -*- coding:utf8 -*-
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import json
from pyproj import CRS, Transformer
import geopandas
import pandas as pd
import matplotlib.colors as mcolors
from shapely.geometry import Point

min_lon, max_lon = 124.4, 131.6
min_lat, max_lat = 33, 38.6

def is_in_boundary(station_data):
  lon = float(station_data['lon'])
  lat = float(station_data['lat'])
  return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

# 1. 이미지에서 추출한 경계값 리스트
# 0.1 미만과 100 초과를 처리하기 위해 실제 컬러바에 표시된 숫자들을 경계값으로 사용
boundaries = [
    0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0,
    70.0, 80.0, 90.0, 100.0
]
# 2. 이미지에서 추출한 색상 리스트 (RGB Hex 코드로 정의)
# ListedColormap은 N개의 경계값에 N-1개의 색상을 매핑하므로,
# under와 over 색상을 제외한 30개의 색상을 추출했습니다.
colors_list = [
    '#ffea6e', # 0.1 - 0.2
    '#ffdc1f', # 0.2 - 0.4
    '#f9cd00', # 0.4 - 0.6
    '#e0b900', # 0.6 - 0.8
    '#ccaa00', # 0.8 - 1.0
    '#69fc69', # 1.0 - 1.5
    '#1ef31e', # 1.5 - 2.0
    '#00d500', # 2.0 - 3.0
    '#00a400', # 3.0 - 4.0
    '#008000', # 4.0 - 5.0
    '#87d9ff', # 5.0 - 6.0
    '#3ec1ff', # 6.0 - 7.0
    '#07abff', # 7.0 - 8.0
    '#008dde', # 8.0 - 9.0
    '#0077b3', # 9.0 - 10.0
    '#b3b4de', # 10.0 - 12.0
    '#8081c7', # 12.0 - 14.0
    '#4c4eb1', # 14.0 - 16.0
    '#1f219d', # 16.0 - 18.0
    '#000390', # 18.0 - 20.0
    '#da87ff', # 20.0 - 25.0
    '#c23eff', # 25.0 - 30.0
    '#ad07ff', # 30.0 - 35.0
    '#9200e4', # 35.0 - 40.0
    '#7f00bf', # 40.0 - 50.0
    '#fa8585', # 50.0 - 60.0
    '#f63e3e', # 60.0 - 70.0
    '#ee0b0b', # 70.0 - 80.0
    '#d50000', # 80.0 - 90.0
    '#bf0000', # 90.0 - 100.0
]

bottom_value = '#eeeeee'
top_value = '#333333'

def create_color_map(boundaries, colors_list, bottom_value, top_value, show_preview=True):
    # Matplotlib ListedColormap을 사용하여 커스텀 컬러맵 생성
    # N = len(boundaries)이므로, boundaries의 개수와 colors_list의 개수를 맞춰야 함.
    cmap = mcolors.ListedColormap(colors_list)

    # 경계값에 따라 Colormap을 정규화하는 BoundaryNorm 생성
    # norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, extend=True)

    # 0.1보다 작은 값과 100보다 큰 값에 대한 색상 설정
    # 이미지에서 0.1 미만은 F0F0F0, 100 초과는 242424로 보임
    cmap.set_under(bottom_value) # 0.1 미만 값
    cmap.set_over(top_value) # 100 초과 값

    if show_preview:
        # 시각화 예시
        # 임의의 데이터 생성
        data = np.random.uniform(4, 6, size=(10, 10))

        # Matplotlib를 사용하여 plot
        fig, ax = plt.subplots(figsize=(8, 10))
        im = ax.imshow(data, cmap=cmap, norm=norm)
        fig.colorbar(im, ax=ax, boundaries=boundaries, extend='both',
                    ticks=boundaries, spacing='proportional')
        ax.set_title("Custom Colormap from User-provided Images")

        plt.show()

    return cmap, norm

# --- 1. 데이터 로드 및 좌표 변환 함수 ---
def load_and_transform_data(json_path: str, transformer: Transformer):
    """
    JSON 파일에서 데이터를 로드하고, 경위도 좌표를 Web Mercator로 변환합니다.
    """
    with open(json_path, encoding='utf-8') as f:
        data_orig = json.load(f)
        data = list(filter(is_in_boundary, data_orig))
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


# --- 한국 boundary masking 생성
def create_kor_boundary_mask(mask, korea_geojson_path: str, XI_WM, YI_WM, lat_cutoff):
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
        # lat_cutoff = 37.7
        _, y_cutoff_wm = transformer.transform(127, lat_cutoff)
        # 3. 위도 기준 필터링
        # Web Mercator의 Y 좌표가 강화도 기준보다 북쪽에 있는 점들만 추출
        points_to_check_gdf = grid_points_gdf[grid_points_gdf['y'] >= y_cutoff_wm].copy()

        korea_boundary_union = korea_boundary_wm.geometry.unary_union
        points_within_korea = points_to_check_gdf.geometry.within(korea_boundary_union)

        # mask = np.zeros_like(ZI_WM, dtype=bool)
        mask_indices = points_to_check_gdf[~points_within_korea]['index'].values
        mask.ravel()[mask_indices] = True
        return mask

        # 음수 또는 아주 작은 값 마스킹 (0.05 미만)
        # mask[ZI_WM < 0.05] = True

        # Z_masked = np.ma.array(ZI_WM, mask=mask)

        # print(f'Number of grid points checked: {len(points_to_check_gdf)} out of {len(grid_points_gdf)}.')

    except Exception as e:
        print(f"Error loading or processing GeoJSON: {e}")
        print("Proceeding without boundary masking. The output will show data outside Korea.")
        return mask  

# --- 3. 마스킹 및 이미지 생성 함수 ---
def create_masked_image(ZI_WM, XI_WM, YI_WM, snow_depths, station_xs_wm, station_ys_wm,
                        image_width_pixels, image_height_pixels,
                        korea_geojson_path: str):
    """
    보간된 데이터에 마스킹을 적용하고, Matplotlib를 사용하여 이미지를 생성합니다.
    """
    try:
        # 대한민국 바운더리 마스킹 적용
        # lat_cutoff = 37.7
        lat_cutoff = 33
        mask = np.zeros_like(ZI_WM, dtype=bool)
        mask_boundary = create_kor_boundary_mask(mask, korea_geojson_path, XI_WM, YI_WM, lat_cutoff)

        # 음수 또는 아주 작은 값 마스킹 (0.05 미만)
        mask_boundary[ZI_WM < 0.05] = True

        Z_masked = np.ma.array(ZI_WM, mask=mask_boundary)
        # print(f'Number of grid points checked: {len(points_to_check_gdf)} out of {len(grid_points_gdf)}.')

    except Exception as e:
        print(f"Error loading or processing GeoJSON: {e}")
        print("Proceeding without boundary masking. The output will show data outside Korea.")
        mask = np.zeros_like(ZI_WM, dtype=bool)
        mask[ZI_WM < 0.05] = True
        Z_masked = np.ma.array(ZI_WM, mask=mask)

    # --- Matplotlib 이미지 생성 및 저장 ---
    print('image_width_pixels', image_width_pixels)
    print('image_height_pixels', image_height_pixels)
    fig = plt.figure(figsize=(25, 24), dpi=100)
    size = fig.get_size_inches() * fig.dpi
    print('Figure size:', size)
    ax = plt.gca()

    # ax.set_aspect('equal')

    # 'cool' 컬러맵 사용
    cmap, norm = create_color_map(boundaries, colors_list, bottom_value, top_value, False)
    # contour = ax.contourf(XI_WM, YI_WM, Z_masked, levels=20, cmap='twilight', extend='neither', alpha=1)
    # contour = ax.contourf(XI_WM, YI_WM, Z_masked, levels=20, cmap=cmap, norm=norm, extend='True', alpha=1)
    contour = ax.contourf(XI_WM, YI_WM, Z_masked, levels=50, cmap=cmap, norm=norm, extend='neither', alpha=1)
    # contour = ax.contourf(XI_WM, YI_WM, Z_masked, levels=50, cmap=cmap, extend='neither', alpha=1)

    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)

    for i in range(len(lons)):
        if snow_depths[i] > 0.0:
            ax.text(station_xs_wm[i], station_ys_wm[i], f'{snow_depths[i]:.1f}',
                    ha='center', va='center', color='red', fontsize=12)

    fig.savefig('snow_depth_mercator_custom_cmap_with_zero.png', bbox_inches='tight', pad_inches=0, transparent=True, facecolor='none')
    plt.close(fig)
    print("Image 'snow_depth_mercator.png' generated.")


# --- 메인 실행 로직 ---
# 주요 프로세스
#
# 적설량이 담긴 json 파일을 읽어 maplibre에 올릴 raster image를 만든다
# 주요 타스크
#
# 1. Maplibre에서 사용할 이미지의 bounds값(4귀퉁이) 결정
# 2. 이미지 생성
#    <첫번째 고비>
#    - 단순하게 json에 있는 위,경도 좌표값을 그대로 x,y 좌표로 사용하는 아이디어를 떠올렸다.
#      예) 이를위한 2가지 step이 있다.
#          4개의 위경도를 꼭지점으로 가지는 grid를 만들고, 그 grid안에 json데이터를 그대로 매핑한다. 
#    - 잘 될 것 같지만, 결과물을 maplibre에 로드해보면 실제와 살짝 어긋난다. (원래 wm에서는 평면이 왜곡된다.)
#    - 그래서 그림을 그릴 때 부터 모든 좌표 숫자를 mercator좌표를 사용한다.
#    => 어떻게 mercator 좌표를 구할 것이냐 => pyproj의 Transformer를 사용하면 쉽다.
#       a. 먼저 json data를 wm으로 변환한다. lon, lat => station_x_wm, station_y_wm (station = 관측소)
#       b. 이미지 네군데 꼭지점 위/경도를 wm으로 변환한다.=> max_lat == max_x_wm
#
#    <두번째 고비>
#    - 모든 좌표를 wm기준으로 얻었으니, 해당 위치에 값들을 넣을 수 있게되었다.
#    - 하지만 300X300 이미지를 채울려면 각 pixel에 대응되는 모든값(90,000)을 가지고 있어야하는데,
#      제한된 측정소 갯수로 훨씬 적은 데이터를 가지고 있다
#    - 여기서 값이 없는 pixel을 채워줄 "보간"이 필요하다.
#    - python에서 보간은 scipy의 interpolate관련 section을 참고하면 되는데 AI는 RBF를 추천했다.
#    - RBF 동작방식은 "가지고 있는 데이터(station x, station y 위치와 값)"를 넘기면 "추정함수"를 리턴한다.
#    - 추정함수는 원하는 좌표(들)을 넘기면, 각 좌표에 추정되는 값을 리턴한다.(값들으 거의 np.array다)
#    - 이제 드문 데이터로 이미지 전체를 채울 수 있는 값들도 얻었다.
#
#    <등고선 그리기 - 작은값 왜곡>
#    - 등고선형태로 그리는 것은 plot.contourf로 할 수 있다.
#    - 그런데, 그냥 그려보면 값이 0인 지역에도 아주 작은 값들이 보정되어 있어 부정확해 보인다.
#    - 그래서 contourf에서 제공하는 mask를 활용할 것이다.
#    - z값이 0.05(작은값)보다 적으면 mask value를 True로 만들어서 투명하게 만든다.
#   
#    <등고선 그리기 - 대한민국 영역만 클리핑>
#    - 위와 비슷한데, 이대로 그리면 북한쪽으로도 보간데이터가 보인다.
#    - 북한영역은 clipping하고 싶은데, 이건 boundary를 사용하면 좋을 것 같다.
#    - 여기는 잘 이해안되는 코드가 많기는 한데, 아무튼 한국 boundary json파일이 있으면 geopandas에서 가능하다고 한다.
#    - AI코드 그대로 사용했다.
# 
#    <등고선에 color map 적용하기>
#    - 적설이니까 contourf 매개변수 cmap에 'cool'로 하면 그럭저럭 잘 나온다.
#    - 하지만 custom color map을 적용해야 기상청 결과물과 비교하기가 쉽다.
#    - 적절한 cmap을 만드는 함수가 필요하고, 값과 color를 어떻게 mapping할지 rule이 필요하다.
#
#    <등고선 그려서 이미지로 저장하기>
#    - plot에서 불필요한 부분(axis, title 등등)은 다 제거하고 그림 부분만 딱 png로 저장한다.
#    - 저장할 때 반드시 alpha(투명도)는 살려야 된다.

if __name__ == '__main__':
    # 설정 변수
    JSON_FILE_PATH = 'SNOW_24H_202507051300_ALL.json'
    # JSON_FILE_PATH = 'SNOW_24H_202507051300.json'
    GEOJSON_FILE_PATH = 'skorea-provinces-2018-geo.json'

    # bbox of image 
    min_lon, max_lon = 124.4, 131.6
    min_lat, max_lat = 33, 38.6
    image_width_pixels = 300
    image_height_pixels = 300
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