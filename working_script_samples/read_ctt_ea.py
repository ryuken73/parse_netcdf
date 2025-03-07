import netCDF4 as nc
import numpy as np
from pyproj import Proj
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# NetCDF 파일 열기
file_path = r"ctps_ea_lc_202502170000.nc"
ds = nc.Dataset(file_path, "r")

# CTT 데이터 읽기 (원래 ushort 값, 자동 스케일링 비활성화)
# ctt_values = ds.variables["CTT"][:, :].data  # raw data로 읽기
ctt_values = ds.variables["CTT"][:]  # raw data로 읽기
dim_y, dim_x = ctt_values.shape  # 2600행, 3000열

# _FillValue 가져오기
fill_value = ds.variables["CTT"]._FillValue  # 65535

# 데이터 타입 확인
print("CTT 데이터 타입:", ctt_values.dtype)

# 투영 정보 가져오기 (LCC)
proj_attrs = ds.variables["gk2a_imager_projection"].__dict__
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

# easting, northing을 위경도로 변환
lon_grid, lat_grid = proj(easting_grid, northing_grid, inverse=True)

# 데이터 리스트 생성: [[lon, lat, ctt_value], ...]
# _FillValue(65535) 제외, step=3으로 샘플링
result = []
step = 5
for y in range(0, dim_y, step):  # 3칸 간격
    for x in range(0, dim_x, step):  # 3칸 간격
        ctt_value = ctt_values[y, x]
        if ctt_value != fill_value:  # _FillValue(65535) 제외
            result.append([
                float(lon_grid[y, x]),  # 경도
                float(lat_grid[y, x]),  # 위도
                int(ctt_value)  # CTT 값 (원래 ushort 값, 21073~30097)
            ])
# 위경도 범위 출력 (디버깅용)
lons = [point[0] for point in result]
lats = [point[1] for point in result]
ctt_values = [point[2] for point in result]
print("Longitude 범위:", min(lons) if lons else "No valid points", "to", max(lons) if lons else "No valid points")
print("Latitude 범위:", min(lats) if lats else "No valid points", "to", max(lats) if lats else "No valid points")
print("CTT 범위:", min(ctt_values) if ctt_values else "No valid points", "to", max(ctt_values) if ctt_values else "No valid points")

# 결과 확인 (처음 5개만 출력)
print("샘플 데이터:", result[:5])

# 결과 확인 (처음 5개만 출력)
print("샘플 데이터:", result[:5])

# JSON 파일로 저장
with open("output_step2_K.json", "w") as f:
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

    scatter = ax.scatter(lons, lats, c=ctt_values, cmap='viridis', s=1, transform=ccrs.PlateCarree())
    plt.colorbar(scatter, label='Cloud Top Temperature (K)')
    plt.title('Geostationary to WGS84 Converted Data (CTT from GK-2A Full Disk)')
    plt.show()

# 파일 닫기
ds.close()