import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer, CRS

# 데이터 파일 설정
data_type = 'bin'
cmp_type = 'hsr'
file_name = r'RDR_CMP_HSP_PUB_202506131900.bin_1'

# 데이터 크기 설정
nx = 2305
ny = 2881

# .bin 파일 읽기
with open(file_name, 'rb') as f:
    bytes = f.read()

rain_rate = np.frombuffer(
    bytes,
    dtype=np.int16,
    offset=1024
).astype(np.float32).reshape(ny, nx)

null_mask = (rain_rate <= -30000)
rain_rate[null_mask] = np.nan
rain_rate /= 100

print(rain_rate)
print(rain_rate.shape)

# 자연스러운 컬러맵 정의 (왼쪽 이미지의 색상 분포 반영)
colors = [
    (0.0, 0.0, 1.0, 1.0),  # 0mm/h: 청색 (하늘색 시작)
    (0.0, 0.5, 0.0, 1.0),  # 2mm/h: 초록색
    (1.0, 1.0, 0.0, 1.0),  # 6mm/h: 노랑색
    (1.0, 0.5, 0.0, 1.0),  # 15mm/h: 주황색
    (1.0, 0.0, 0.0, 1.0),  # 30mm/h: 빨간색
    (0.5, 0.0, 0.5, 1.0),  # 70mm/h: 자주색
]
n_bins = 256  # 부드러운 전환을 위한 색상 단계 수
colormap_rain = LinearSegmentedColormap.from_list("rain_gradient", colors, N=n_bins)

# 데이터 분포에 맞춘 정규화
vmax = np.nanmax(rain_rate) if np.nanmax(rain_rate) > 0 else 150
norm = Normalize(vmin=0, vmax=vmax)

# 컬러 배열 생성
colored_array = norm(rain_rate)
colored_array = colormap_rain(colored_array)
colored_array[null_mask] = [0, 0, 0, 0]  # 결측치는 투명으로
colored_array = (colored_array * 255).astype(np.uint8)

# Matplotlib으로 시각화 (원본 데이터)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title('Colored array')
ax.set_facecolor('#cccccc')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0)
im = ax.imshow(colored_array, origin='lower')
cbar = fig.colorbar(im, cax=cax, ticks=[0, 2, 6, 15, 30, 70], label='mm/h')
cbar.ax.tick_params(labelsize=8)

# 좌표계 변환 준비
source_width = nx
source_height = ny
source_center_x = 1121
source_center_y = 1681
source_resolution = 500

source_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
source_transform = Affine.scale(source_resolution, source_resolution) * Affine.translation(-source_center_x, -source_center_y)

source_bounds = {
    'left': -source_center_x * source_resolution,
    'bottom': (source_height - source_center_y) * source_resolution,
    'right': (source_width - source_center_x) * source_resolution,
    'top': -source_center_y * source_resolution
}

# Web Mercator로 변환
dest_transform, dest_width, dest_height = calculate_default_transform(
    src_crs=source_crs,
    dst_crs='EPSG:3857',
    width=source_width,
    height=source_height,
    **source_bounds,
)

converted_array = np.ones((dest_height, dest_width, 4), dtype=np.uint8)

for i in range(4):
    reproject(
        source=colored_array[:, :, i],
        destination=converted_array[:, :, i],
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=dest_transform,
        dst_crs='EPSG:3857',
        resampling=Resampling.bilinear,  # 부드러운 전환을 위해 bilinear 사용
    )

# 변환된 이미지 시각화
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title('Converted array')
ax.set_facecolor('#cccccc')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0)
im = ax.imshow(converted_array)
print(converted_array.shape)
cbar = fig.colorbar(im, cax=cax, ticks=[0, 2, 6, 15, 30, 70], label='mm/h')
cbar.ax.tick_params(labelsize=8)

# Web Mercator 및 WGS84 좌표계 정의
web_mercator_crs = CRS.from_epsg("3857")  # Web Mercator
wgs84_crs = CRS.from_epsg("4326")         # WGS84

# Web Mercator bounds 계산
top_left = dest_transform * (0, 0)
top_right = dest_transform * (dest_width, 0)
bottom_right = dest_transform * (dest_width, dest_height)
bottom_left = dest_transform * (0, dest_height)

transformer_mercator_to_wgs84 = Transformer.from_crs(web_mercator_crs, wgs84_crs, always_xy=True)
lon_top_left, lat_top_left = transformer_mercator_to_wgs84.transform(top_left[0], top_left[1])
lon_top_right, lat_top_right = transformer_mercator_to_wgs84.transform(top_right[0], top_right[1])
lon_bottom_right, lat_bottom_right = transformer_mercator_to_wgs84.transform(bottom_right[0], bottom_right[1])
lon_bottom_left, lat_bottom_left = transformer_mercator_to_wgs84.transform(bottom_left[0], bottom_left[1])

# Mapbox용 bounds (좌상단, 우상단, 우하단, 좌하단)
mapbox_bounds = [
    [lon_top_left, lat_top_left],    # 좌상단
    [lon_top_right, lat_top_right],  # 우상단
    [lon_bottom_right, lat_bottom_right],  # 우하단
    [lon_bottom_left, lat_bottom_left]     # 좌하단
]

print("Mapbox bounds (WGS84 coordinates):", mapbox_bounds)

# 변환된 이미지 저장
image = Image.fromarray(converted_array.astype(np.uint8), 'RGBA')
image.save('2506131900_v3.png')