import netCDF4 as nc
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# NetCDF 파일 경로
file_path = 'ir105_fd_ge_202502170000.nc'

# NetCDF 파일 열기
ds = nc.Dataset(file_path, 'r')

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

# ground_sample_distance 조정 (2km로 가정)
ground_sample_distance = 2000.0  # 자료집에 따른 2km 해상도
resolution_rad = ground_sample_distance / a  # 2km를 라디안으로 변환 (약 0.000313 라디안)
print(f"Adjusted ground_sample_distance: {ground_sample_distance} m, resolution_rad: {resolution_rad} rad")

# GEOS 좌표를 WGS84 위도/경도로 변환하는 함수 (NetCDF sub_lon 적용)
def latlon_from_geos(Line, Column, sub_lon, coff, cfac, loff, lfac):
    degtorad = 3.14159265358979 / 180.0
    # sub_lon = 128.2  # 고정값 제거
    # sub_lon = sub_lon * degtorad  # NetCDF 값 사용
    x = degtorad * ((Column - coff) * 2**16 / cfac)
    y = degtorad * ((dim_y - Line - 1 - loff) * 2**16 / lfac)  # 행 좌표 뒤집기
    Sd = np.sqrt((42164.0 * np.cos(x) * np.cos(y))**2 - (np.cos(y)**2 + 1.006739501 * np.sin(y)**2) * 1737122264)
    Sn = (42164.0 * np.cos(x) * np.cos(y) - Sd) / (np.cos(y)**2 + 1.006739501 * np.sin(y)**2)
    S1 = 42164.0 - (Sn * np.cos(x) * np.cos(y))
    S2 = Sn * (np.sin(x) * np.cos(y))
    S3 = -Sn * np.sin(y)
    Sxy = np.sqrt((S1 * S1) + (S2 * S2))
    nlon = (np.arctan(S2 / S1) + sub_lon) / degtorad  # NetCDF sub_lon 사용
    nlat = np.arctan((1.006739501 * S3) / Sxy) / degtorad
    return nlon, nlat

# 결과 리스트 초기화
result = []
step = 10  # 샘플링 간격 (필요에 따라 조정 가능)

# 각 픽셀에 대해 위도/경도 변환 및 데이터 추출
for y in range(0, dim_y, step):
    for x in range(0, dim_x, step):
        lon, lat = latlon_from_geos(y, x, sub_lon, coff, cfac, loff, lfac)
        if lon is not None and lat is not None:  # 유효한 좌표만 포함
            # 동아시아 범위(90°E~150°E, -10°~50°N)로 필터링
            if 90 <= lon <= 150 and -10 <= lat <= 50:
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
    ax.set_extent([90, 150, -10, 50], crs=ccrs.PlateCarree())  # 동아시아 범위
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
    ax.add_feature(cfeature.LAND, facecolor='lightgreen')

    scatter = ax.scatter(lons, lats, c=[point[2] for point in result], cmap='viridis', s=1, transform=ccrs.PlateCarree())
    plt.colorbar(scatter, label='Pixel Value')
    plt.title('Geostationary to WGS84 Converted Data (East Asia)')
    plt.show()

# 파일 닫기
ds.close()