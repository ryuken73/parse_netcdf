# ir105 fd nc파일 입력
# np vector 연산으로 빠르게 np.array(lon, lat, value)생성 후
# np vector 연산기반으로 이미지를 생성하는 generate_image_from_data_fast를
# 사용하여 빠르게 이미지를 만듬
# step1, 3192X3192가 소요시간(5분) 대비 품질이 괜찮음


import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
from pyproj import Transformer, CRS
from mk_image_mercator_geo_color import generate_image_from_data
# from mk_image_faster_with_vector import generate_image_from_data_fast
from mk_image_faster_with_vector_last import generate_image_from_data_fast

step = 1

def load_conversion_table(file_path):
    conversion_table = []
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 2:
                conversion_table.append(float(values[1]))
    return np.array(conversion_table)

# NetCDF 파일 열기 및 처리
file_path = 'gk2a_ami_le1b_ir105_fd020ge_202503232320_202503240820.nc'
ds = nc.Dataset(file_path, format='NETCDF4')
image_pixel_values = ds.variables['image_pixel_values'][:]
# 원본 코드에서 이미지 데이터 크기 확인
dim_y, dim_x = image_pixel_values.shape
print("image_pixel_values shape:", image_pixel_values.shape)

# 제공된 NetCDF 파일에서 위경도 테이블 로드
latlon_ds = nc.Dataset('C:/Users/USER/Downloads/gk2a_ami_fd020ge_latlon.nc', format='NETCDF4')
lat = latlon_ds.variables['lat'][:]
lon = latlon_ds.variables['lon'][:]
latlon_ds.close()

# step을 적용한 인덱스 생성
y_indices = np.arange(0, dim_y, step)
x_indices = np.arange(0, dim_x, step)
Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')

# 인덱스를 사용해 위경도 배열 샘플링
sampled_lat = lat[Y, X]
sampled_lon = lon[Y, X]

# 경도 변환: -180~0도 사이 값을 180~360도로 변환
valid_lon_mask = (sampled_lon >= -180) & (sampled_lon <= 180)  # 유효한 경도 값만 선택
adjusted_lon = np.where(valid_lon_mask, sampled_lon, np.nan)  # 비정상 값은 NaN으로 처리
adjusted_lon = np.where(adjusted_lon < 0, adjusted_lon + 360, adjusted_lon)  # -180~0도를 180~360도로 변환

# 위경도 범위 마스크 적용: 경도 범위를 50~230도로 확장
latlon_mask = (sampled_lat >= -80) & (sampled_lat <= 80) & (adjusted_lon >= 30) & (adjusted_lon <= 230)
final_mask = latlon_mask

# 변환 테이블 적용
values = image_pixel_values[Y, X]
print('type of values', values)
conversion_file = "ir105_conversion_c.txt"
lut = load_conversion_table(conversion_file)

# 마스크를 사용한 안전한 변환
mask = (values >= 0) & (values < len(lut))
converted_values = np.full_like(values, -9999, dtype=float)
converted_values[mask] = lut[values[mask]]

print("Max value in converted_values:", np.max(converted_values))
print("Min value in converted_values:", np.min(converted_values))

# 결과 배열 생성
result = np.column_stack([
    adjusted_lon[final_mask].astype(float),
    sampled_lat[final_mask].astype(float),
    converted_values[final_mask].astype(float)
])

# 디버깅 출력: 경도 데이터 확인
lons = result[:, 0]
lats = result[:, 1]
print("Longitude 범위 (result):", np.min(lons) if lons.size > 0 else "No valid points", 
      "to", np.max(lons) if lons.size > 0 else "No valid points")
print("Latitude 범위 (result):", np.min(lats) if lats.size > 0 else "No valid points", 
      "to", np.max(lats) if lats.size > 0 else "No valid points")
print("180도를 넘는 경도 값 개수:", np.sum(lons > 180), "개")
print("샘플 데이터:", result[:5])

# 경도 범위와 이미지 크기 비율 조정
lon_range = 230 - 30  # 경도 범위: 180도
lat_range = 80 - (-80)  # 위도 범위: 160도
aspect_ratio = lon_range / lat_range  # 경도:위도 비율 = 180/160 = 1.125
image_width = 1500
image_height = int(image_width / aspect_ratio)  # 비율에 맞춰 높이 조정

data_list = result.tolist()

# PNG 이미지 생성: bounds와 이미지 크기 조정
output_path = "output_image_fast_w_vector.png"
bounds = [40, -80, 230, 80]  # 경도 범위를 50~230도로 유지
print(f"이미지 크기 (width, height): ({image_width}, {image_height})")
generate_image_from_data_fast(result, output_path, image_size=(image_width, image_height), bounds=bounds)

ds.close()