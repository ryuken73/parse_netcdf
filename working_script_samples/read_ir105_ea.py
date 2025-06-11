import netCDF4 as nc
import numpy as np
from pyproj import Proj
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

file_path = 'gk2a_ami_le1b_ir105_ea020lc_202502281500.nc'

ds = nc.Dataset(file_path, format='NETCDF4')


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
step = 1 
result = []
for y in range(0, dim_y, step):
    for x in range(0, dim_x, step):
        result.append([
            float(lon_grid[y, x]),  # 경도
            float(lat_grid[y, x]),  # 위도
            int(image_pixel_values[y, x])  # 픽셀 값 (ushort -> int로 변환)
        ])

# 위경도 범위 출력 (디버깅용)
lons = [point[0] for point in result]
lats = [point[1] for point in result]
img_pix_values = [point[2] for point in result]
print("Longitude 범위:", min(lons) if lons else "No valid points", "to", max(lons) if lons else "No valid points")
print("Latitude 범위:", min(lats) if lats else "No valid points", "to", max(lats) if lats else "No valid points")
print("image pixel values 범위:", min(img_pix_values) if img_pix_values else "No valid points", "to", max(img_pix_values) if img_pix_values else "No valid points")

# 결과 확인 (처음 5개만 출력)
print("샘플 데이터:", result[:5])

# JSON 파일로 저장
with open(f"ir105_ea_lc_202502170000_step{step}.json", "w") as f:
    json.dump(result, f)
print(f"데이터가 output.json에 저장되었습니다. 총 {len(result)}개 포인트")

# Matplotlib과 Cartopy를 사용한 시각화
if lons and lats:
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([70, 180, 0, 80], crs=ccrs.PlateCarree())  # 남/북 범위 확장
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
    ax.add_feature(cfeature.LAND, facecolor='lightgreen')

    scatter = ax.scatter(lons, lats, c=img_pix_values, cmap='viridis', s=1, transform=ccrs.PlateCarree())
    plt.colorbar(scatter, label='IR105 pixel values')
    plt.title('IR105 EA')
    plt.show()

# 파일 닫기
ds.close()