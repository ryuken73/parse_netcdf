import netCDF4 as nc
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# NetCDF 파일 경로
file_path = 'gk2a_ami_le1b_ir105_fd020ge_202503232320_202503240820.nc'

# NetCDF 파일 열기
ds = nc.Dataset(file_path, format='NETCDF4')

# 이미지 픽셀 값과 차원 가져오기
image_pixel_values = ds.variables['image_pixel_values'][:]
dim_y, dim_x = image_pixel_values.shape
print("image_pixel_values shape:", image_pixel_values.shape)

# GEOS 투영 파라미터 (NetCDF 글로벌 속성에서 추출)
sub_lon = ds.getncattr('sub_longitude')  # 위성 경도 (라디안, 약 128.2°E)
H = ds.getncattr('nominal_satellite_height')  # 위성 높이 (미터)
a = ds.getncattr('earth_equatorial_radius')  # 지구 적도 반지름 (미터)
b = ds.getncattr('earth_polar_radius')  # 지구 극 반지름 (미터)
cfac = ds.getncattr('cfac')  # 열 방향 계수
lfac = ds.getncattr('lfac')  # 행 방향 계수 (음수로 정의)
coff = ds.getncattr('coff')  # 열 중심
loff = ds.getncattr('loff')  # 행 중심

# ground_sample_distance 조정 (2km로 가정, 기본 스케일링 유지)
ground_sample_distance = 2000.0  # 자료집에 따른 2km 해상도
resolution_factor = 1.0  # resolution_factor 제거, cfac, lfac에 의존
print(f"Adjusted ground_sample_distance: {ground_sample_distance} m, resolution_factor: {resolution_factor}")

# GEOS 좌표를 WGS84 위도/경도로 변환하는 함수
def latlon_from_geos(Line, Column, sub_lon, coff, cfac, loff, lfac):
    # print(Line, Column, sub_lon, coff, cfac, loff, lfac)
    degtorad = 3.14159265358979 / 180.0
    x = degtorad * ((Column - coff) * 2**16 / cfac)  # resolution_factor 제거
    y = degtorad * ((dim_y - Line - 1 - loff) * 2**16 / lfac)  # 행 좌표 뒤집기
    # 일부 로그만 출력 (처음 50x50 범위 내에서만)
    # print(f"Line={Line}, Column={Column}, x={x}, y={y}")
    # if Line < 50 and Column < 50:
    #     print(f"x = {x}, y = {y} at (Line={Line}, Column={Column})")
    Sd = np.sqrt((42164.0 * np.cos(x) * np.cos(y))**2 - (np.cos(y)**2 + 1.006739501 * np.sin(y)**2) * 1737122264)
    if Sd < 0:  # 무효값 방지 및 디버깅
        print(f"Sd negative: {Sd} at (x={x}, y={y}, Line={Line}, Column={Column})")
        return None, None
    Sn = (42164.0 * np.cos(x) * np.cos(y) - Sd) / (np.cos(y)**2 + 1.006739501 * np.sin(y)**2)
    S1 = 42164.0 - (Sn * np.cos(x) * np.cos(y))
    S2 = Sn * (np.sin(x) * np.cos(y))
    S3 = -Sn * np.sin(y)
    Sxy = np.sqrt((S1 * S1) + (S2 * S2))
    nlon = (np.arctan(S2 / S1) + sub_lon) / degtorad
    nlat = np.arctan((1.006739501 * S3) / Sxy) / degtorad
    return nlon, nlat

# 결과 리스트 초기화
result = []
step = 1  # 데이터 밀도 유지

# 각 픽셀에 대해 위도/경도 변환 및 데이터 추출
for y in range(0, dim_y, step):
    for x in range(0, dim_x, step):
        lon, lat = latlon_from_geos(y, x, sub_lon, coff, cfac, loff, lfac)
        if lon is not None and lat is not None:  # 유효한 좌표만 포함
            if -80 <= lat <= 80 and -180 <= lon <= 190:  # 남/북 범위 확장
                result.append([
                    float(lon),  # 경도
                    float(lat),  # 위도
                    int(image_pixel_values[y, x])  # 픽셀 값 (ushort -> int)
                ])

# 위경도 범위 출력 (디버깅용)
lons = [point[0] for point in result]
lats = [point[1] for point in result]
print("Longitude 범위:", min(lons) if lons else "No valid points", "to", max(lons) if lons else "No valid points")
print("Latitude 범위:", min(lats) if lats else "No valid points", "to", max(lats) if lats else "No valid points")

# 결과 확인 (처음 5개만 출력)
print("샘플 데이터:", result[:5])

# JSON 파일로 저장
output_filename = f"output_ir105_fd_ge_step{step}.json"
with open(output_filename, "w") as f:
    json.dump(result, f)
print(f"데이터가 {output_filename}에 저장되었습니다. 총 {len(result)}개 포인트")

# Matplotlib과 Cartopy를 사용한 시각화
if lons and lats:
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 190, -80, 80], crs=ccrs.PlateCarree())  # 남/북 범위 확장
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
    ax.add_feature(cfeature.LAND, facecolor='lightgreen')

    scatter = ax.scatter(lons, lats, c=[point[2] for point in result], cmap='viridis', s=1, transform=ccrs.PlateCarree())
    plt.colorbar(scatter, label='Pixel Value')
    plt.title('Geostationary to WGS84 Converted Data (GK-2A Full Disk)')
    plt.show()

# 파일 닫기
ds.close()