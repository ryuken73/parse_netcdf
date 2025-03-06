import netCDF4 as nc
import numpy as np
from pyproj import Proj
import json

file_path = 'ir105_ea_lc_202502170000.nc'

ds = nc.Dataset(file_path, 'r')


# lat = dataset.variables['lat'][:]

image_pixel_values = ds.variables['image_pixel_values'][:]
dim_y, dim_x = image_pixel_values.shape
print("image_pixel_values", image_pixel_values)

# 투영 정보 가져오기 (LCC)
proj_attrs = ds.__dict__
print("proj_attrs", proj_attrs)
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

# 각 픽셀의 easting, northing 좌표 계산
easting = np.linspace(upper_left_easting, upper_left_easting + pixel_size * (dim_x - 1), dim_x)
northing = np.linspace(upper_left_northing, upper_left_northing - pixel_size * (dim_y - 1), dim_y)
easting_grid, northing_grid = np.meshgrid(easting, northing)

lon_grid, lat_grid = proj(easting_grid, northing_grid, inverse=True)

# 데이터 리스트 생성: [[lon, lat, image_pixel_value], ...]
step = 5 
result = []
for y in range(0, dim_y, step):
    for x in range(0, dim_x, step):
        result.append([
            float(lon_grid[y, x]),  # 경도
            float(lat_grid[y, x]),  # 위도
            int(image_pixel_values[y, x])  # 픽셀 값 (ushort -> int로 변환)
        ])

# 결과 확인 (처음 5개만 출력)
print("샘플 데이터:", result[:5])

# JSON 파일로 저장
with open("ir105_ea_lc_202502170000_step10.json", "w") as f:
    json.dump(result, f)
print(f"데이터가 output.json에 저장되었습니다. 총 {len(result)}개 포인트")

# 파일 닫기
ds.close()