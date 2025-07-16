import os

data_type = 'bin'
cmp_type = 'hsr'

file_name = r'c:\\temp\\RDR_CMP_HSP_PUB_202506131900.bin_1'


import numpy as np


nx = 2305
ny = 2881

with open(file_name, 'rb') as f:
    bytes = f.read()

rain_rate = np.frombuffer(
    bytes,
    dtype=np.int16,
    offset=1024
).astype(np.float32).reshape(ny, nx)

#print(rain_rate)

null_mask = (rain_rate <= -30000)

rain_rate[null_mask] = np.nan

rain_rate /= 100

print(rain_rate)
print(rain_rate.shape)

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize

colormap_rain = ListedColormap(np.array([
    [255,255,255, 0], [0, 200, 255, 255], [0, 155, 245, 255], [0, 74, 245, 255],                # 하늘색
    [0, 255, 0, 255], [0, 190, 0, 255], [0, 140, 0, 255], [0, 90, 0, 255],                          # 초록색
    [255, 255, 0, 255], [255, 220, 31, 255], [249, 205, 0, 255], [224, 185, 0, 255], [204, 170, 0, 255], # 노랑색
    [255, 102, 0, 255], [255, 50, 0, 255], [210, 0, 0, 255], [180, 0, 0, 255],                      # 빨간색
    [224, 169, 255, 255], [201, 105, 255, 255], [179, 41, 255, 255], [147, 0, 228, 255],            # 보라색
    [179, 180, 222, 255], [76, 78, 177, 255], [0, 3, 144, 255], [51, 51, 51, 255]                   # 파란색
]) / 255)

colormap_rain.set_bad([0, 0, 0, 0])

bounds = np.array([
    0, 0.1, 0.5, 1, # 하늘색
    2, 3, 4, 5,     # 초록색
    6, 7, 8, 9, 10, # 노랑색
    15, 20, 25, 30, # 빨간색
    40, 50, 60, 70, # 보라색
    90, 110, 150    # 파란색
])

# 색상 개수와 값 범위를 서로 맞춥니다.
norm = BoundaryNorm(boundaries=bounds, ncolors=len(colormap_rain.colors))

print(norm)

from mpl_toolkits.axes_grid1 import make_axes_locatable

colored_array = BoundaryNorm(
    boundaries=bounds,
    ncolors=len(colormap_rain.colors)
)(rain_rate)
colored_array = Normalize(
    0, len(colormap_rain.colors)
)(colored_array)
colored_array[null_mask] = np.nan
colored_array = (colormap_rain(colored_array) * 255).astype(np.uint8)

# 분포도에 표시되는 범위 값을 2칸 간격으로 구합니다.
ticks = bounds[:]

# 분포도를 그릴 크기를 지정합니다.
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# 분포도의 제목을 지정합니다.
ax.set_title('Colored array')

# 배경색을 회색으로 지정합니다.
ax.set_facecolor('#cccccc')

# 색상표의 크기와 위치를 조절합니다.
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0)

# 배열을 해당 크기에 맞춰 그립니다.
im = ax.imshow(colored_array, origin='lower', cmap=colormap_rain, norm=norm)

# 색상표에 표시될 글자 크기 및 제목을 설정합니다.
cbar = fig.colorbar(im, cax=cax, ticks=ticks)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_title('mm/h', fontsize=8)


# 좌표계 변환을 위한 변환 행렬
from rasterio.transform import Affine

# 좌표계 변환을 위해 필요한 함수
from rasterio.warp import calculate_default_transform, reproject, Resampling

# 레이더 자료의 너비와 높이, 중심점 좌표와 공간 해상도를 m단위로 정의합니다.
source_width = nx
source_height = ny
source_center_x = 1121
source_center_y = 1681
source_resolution = 500

# 변환 전의 좌표계를 proj.4 형태의 문자열로 정의합니다.
# 투영법(Projection)은 LCC이며 LCC 좌표계 정의에 필요한 위도 선 2개(lat_1, lat_2)와
# 중심 위도, 경도(lat_0, lon_0)을 정의합니다.
# 좌표계의 x, y는 각각 오른쪽, 위로 증가하는 방향을 가지고(False easting, False northing)
# 지구 타원체를 WGS84로 정의합니다.
# 마지막으로 좌표계가 사용하는 단위는 미터(m)로 정의합니다.
source_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

# 이미지(배열)의 행과 열이 LCC 좌표계에서의 좌표로 변환되기 위한 변환 행렬(Affine Transform Matrix)을 정의합니다.
# 변환 행렬은 먼저 이미지의 중심점을 (0, 0) 위치로 이동시킨 뒤(Translation)
# 각 이미지 픽셀이 실제로 500m를 나타내므로 크기를 x, y로 500배 키워주는(Scale) 역할을 합니다.
# 이때, 행렬의 곱하는 순서에 유의합니다.(AB != BA)
source_transform = Affine.scale(source_resolution, source_resolution) * Affine.translation(-source_center_x, -source_center_y)

# 변환 행렬을 거친 이미지가 나타내는 경계를 정의합니다.
source_bounds = {
    'left': -source_center_x * source_resolution,
    'bottom': (source_height - source_center_y) * source_resolution,
    'right': (source_width - source_center_x) * source_resolution,
    'top': -source_center_y * source_resolution
}

# 변환 후 이미지의 변환 행렬과 너비와 높이를 계산합니다.
dest_transform, dest_width, dest_height = calculate_default_transform(
    src_crs=source_crs,
    dst_crs='EPSG:3857',
    width=source_width,
    height=source_height,
    **source_bounds,
)

# 변환 후의 이미지가 담길 비어있는 배열을 정의합니다.
converted_array = np.ones((dest_height, dest_width, 4), dtype=np.uint8)

# RGBA 각 채널에 대해 좌표계 변환을 수행합니다.
# 사용하는 resampling 기법으로 가까운 값을 선택하는 nearest를 선택합니다.
for i in range(4):
    reproject(
        source=colored_array[:, :, i],
        destination=converted_array[:, :, i],
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=dest_transform,
        dst_crs='EPSG:3857',
        resampling=Resampling.nearest,
    )

# 변환된 이미지를 분포도로 표출합니다.
# 이전 분포도와 다르게 원점의 위치, 너비, 높이가 달라졌음을 확인합니다.
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# 분포도의 제목을 지정합니다.
ax.set_title('Converted array')

# 배경색을 회색으로 지정합니다.
ax.set_facecolor('#cccccc')

# 색상표의 크기와 위치를 조절합니다.
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0)

# 배열을 표출하며 정의한 색상표와 범위를 지정합니다.
im = ax.imshow(converted_array, cmap=colormap_rain, norm=norm)
print(converted_array.shape)

# 색상표에 표시될 글자 크기 및 제목을 설정합니다.
cbar = fig.colorbar(im, cax=cax, ticks=ticks)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_title('mm/h', fontsize=8)

print(converted_array)


from PIL import Image
image = Image.fromarray(converted_array.astype(np.uint8), 'RGBA')
image.save('2506131900_1.png')

# 그린 이미지를 표출합니다.
plt.show()


import folium   # 지도로 표출하기 위한 라이브러리
import branca.colormap as cm    # 지도에 색상표를 표출하기 위한 색상표 모듈

# 지도에 마우스 위치를 표출하는 플러그인
from folium.plugins import MousePosition

# 점끼리의 좌표계 변환을 위한 transformer
from pyproj.transformer import Transformer

# folium을 이용한 좌표는 경도, 위도만 입력 가능하므로
# EPSG:3857 좌표계에서 EPSG:4326 좌표계로의 변환하는 transformer를 정의합니다.
degree_tranformer = Transformer.from_crs('EPSG:3857', 'EPSG:4326')

# 너비, 높이 크기가 800인 지도 생성
fig = folium.Figure(width=800, height=800)

# 지도의 중심과 나타낼 영역을 위·경도로 입력합니다.
map = folium.Map(
    location=[38, 126],
    zoom_start=6,
    min_zoom=6,
    min_lat=28,
    max_lat=43,
    min_lon=116,
    max_lon=135,
    max_bounds=True,
).add_to(fig)

# 지도에 색상표를 추가합니다.
map.add_child(cm.StepColormap(
    [tuple(i) for i in colormap_rain.colors],
    vmin=bounds[0], vmax=bounds[-1], tick_labels=[], caption='mm/h'
))

# 변환된 이미지를 지도 위에 겹쳐 그립니다.
# 불투명도를 0.4로 정하고
# 이미지가 그려질 범위를 경도, 위도로 입력합니다.
folium.raster_layers.ImageOverlay(
    image=converted_array,
    name='rain_rate',
    opacity=0.4,
    bounds=[
        degree_tranformer.transform(*dest_transform.__mul__((0, dest_height))),
        degree_tranformer.transform(*dest_transform.__mul__((dest_width, 0)))
    ],
    zindex=1,
).add_to(map)

# 지도에 마우스 위치를 표출하는 플러그인과 layer를 조절하는 버튼을 추가합니다.
MousePosition().add_to(map)
folium.LayerControl().add_to(map)

# 지도를 표출합니다.
fig



#print(colormap_rain)
#data = np.random.rand(100, 100) * 3
#img = plt.imshow(data, cmap=colormap_rain)
#plt.colorbar(img)
#plt.savefig('bar.png')
