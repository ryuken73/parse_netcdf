import netCDF4 as nc
import numpy as np
from pyproj import Proj
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mk_image_faster_with_vector import generate_image_from_data_fast

# NetCDF 파일 경로
file_path = 'gk2a_ami_le1b_ir105_ko020lc_202505220730_202505221630.nc'

# 파일 열기
ds = nc.Dataset(file_path, format='NETCDF4')

def load_conversion_table(file_path):
    conversion_table = []
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 2:
                conversion_table.append(float(values[1]))
    return np.array(conversion_table)

# 이미지 픽셀 값과 차원 가져오기
image_pixel_values = ds.variables['image_pixel_values'][:]
dim_y, dim_x = image_pixel_values.shape
print("image_pixel_values shape:", image_pixel_values.shape)

# 투영 정보 가져오기 (LCC)
proj_attrs = ds.__dict__
central_meridian = proj_attrs["central_meridian"]  # 126.0
standard_parallel1 = proj_attrs["standard_parallel1"]  # 30.0
standard_parallel2 = proj_attrs["standard_parallel2"]  # 60.0
origin_latitude = proj_attrs["origin_latitude"]  # 38.0
false_easting = proj_attrs["false_easting"]  # 0.0
false_northing = proj_attrs["false_northing"]  # 0.0
pixel_size = proj_attrs["pixel_size"]  # 2000.0 (미터 단위)
upper_left_easting = proj_attrs["upper_left_easting"]  # -2999000.0
upper_left_northing = proj_attrs["upper_left_northing"]  # 2599000.0

# LCC 투영 정의
proj = Proj(
    proj="lcc",
    lat_1=standard_parallel1,
    lat_2=standard_parallel2,
    lat_0=origin_latitude,
    lon_0=central_meridian,
    x_0=false_easting,
    y_0=false_northing,
    units="m",
    ellps="WGS84"
)

# 각 픽셀의 easting, northing 좌표 계산 (이미 벡터화됨)
easting = np.linspace(upper_left_easting, upper_left_easting + pixel_size * (dim_x - 1), dim_x)
northing = np.linspace(upper_left_northing, upper_left_northing - pixel_size * (dim_y - 1), dim_y)
easting_grid, northing_grid = np.meshgrid(easting, northing)

# LCC -> WGS84 변환 (벡터화)
lon_grid, lat_grid = proj(easting_grid, northing_grid, inverse=True)

# 샘플링 인덱스 생성
step = 1
y_indices = np.arange(0, dim_y, step)
x_indices = np.arange(0, dim_x, step)
Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')

# 벡터화된 데이터 추출
lons = lon_grid[Y, X].astype(float)  # 경도
lats = lat_grid[Y, X].astype(float)  # 위도
values = image_pixel_values[Y, X].astype(int)  # 픽셀 값
print('type of values', values)
conversion_file = "ir105_conversion_c.txt"
lut = load_conversion_table(conversion_file)

# 마스크를 사용한 안전한 변환
mask = (values >= 0) & (values < len(lut))
converted_values = np.full_like(values, -9999, dtype=float)  # 기본값 -9999
converted_values[mask] = lut[values[mask]]  # 유효한 값만 변환

print("Max value in converted_values:", np.max(converted_values))
print("Min value in converted_values:", np.min(converted_values))


# 결과 배열 생성
result = np.column_stack([lons.flatten(), lats.flatten(), converted_values.flatten()])

# 위경도 및 픽셀 값 범위 출력 (디버깅용)
print("Longitude 범위:", np.min(lons) if lons.size > 0 else "No valid points",
      "to", np.max(lons) if lons.size > 0 else "No valid points")
print("Latitude 범위:", np.min(lats) if lats.size > 0 else "No valid points",
      "to", np.max(lats) if lats.size > 0 else "No valid points")
print("Image pixel values 범위:", np.min(values) if values.size > 0 else "No valid points",
      "to", np.max(values) if values.size > 0 else "No valid points")

# 결과 확인 (처음 5개만 출력)
print("샘플 데이터:", result[:5])

# JSON 파일로 저장
# output_filename = f"ir105_ea_lc_202502170000_step{step}.json"
# with open(output_filename, "w") as f:
#     json.dump(result.tolist(), f)  # NumPy 배열을 리스트로 변환
# print(f"데이터가 {output_filename}에 저장되었습니다. 총 {len(result)}개 포인트")

# 파일 닫기

# PNG 파일로 저장
output_path = "output_image_fast_w_vector_ko_color.png"
# ea_bounds = [76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447]
ko_bounds = [113.99641, 29.312252, 138.003582, 45.728965]
generate_image_from_data_fast(result, output_path, image_size=(800, 800), bounds=ko_bounds)
# generate_image_from_data_fast(result, output_path, image_size=(900, 900), bounds=ko_bounds)

# ds.close()

# if lons.size > 0 and lats.size > 0:
#     plt.figure(figsize=(10, 8))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE)
#     scatter = ax.scatter(lons, lats, c=values, cmap='viridis', s=1)
#     plt.colorbar(scatter, label='Pixel Value')
#     plt.show()