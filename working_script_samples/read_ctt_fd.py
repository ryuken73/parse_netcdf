import netCDF4 as nc
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# NetCDF 파일 경로
file_path = 'ctps_fd_ge_202502170000.nc'

# NetCDF 파일 열기
ds = nc.Dataset(file_path, 'r')

# 이미지 픽셀 값과 차원 가져오기
ctt_raw = ds.variables['CTT'][:]  # CTT 데이터 (raw 값)
dim_y, dim_x = ctt_raw.shape
print("CTT shape:", ctt_raw.shape)

# GEOS 투영 파라미터 (gk2a_imager_projection 변수에서 추출)
projection = ds.variables['gk2a_imager_projection']
sub_lon_deg = projection.longitude_of_projection_origin  # 위성 경도 (도 단위)
sub_lon = sub_lon_deg * math.pi / 180.0  # 도를 라디안으로 변환
H = projection.perspective_point_height / 1000.0     # 위성 높이 (km 단위로 변환)
a = projection.semi_major_axis                       # 지구 적도 반지름 (미터)
b = projection.semi_minor_axis                       # 지구 극 반지름 (미터)
cfac = projection.column_scale_factor                # 열 방향 계수
lfac = projection.line_scale_factor                  # 행 방향 계수
coff = projection.column_offset                      # 열 중심
loff = projection.line_offset                        # 행 중심

print(f"sub_lon (deg): {sub_lon_deg}, sub_lon (rad): {sub_lon}, H: {H} km, a: {a} m, b: {b} m, cfac: {cfac}, lfac: {lfac}, coff: {coff}, loff: {loff}")

# ground_sample_distance 조정 (2km로 가정, 기본 스케일링 유지)
ground_sample_distance = 2000.0
resolution_factor = 1.0  # resolution_factor 제거, cfac, lfac에 의존
print(f"Adjusted ground_sample_distance: {ground_sample_distance} m, resolution_factor: {resolution_factor}")

# GEOS 좌표를 WGS84 위도/경도로 변환하는 함수
def latlon_from_geos(Line, Column, sub_lon, coff, cfac, loff, lfac):
    degtorad = 3.14159265358979 / 180.0
    x = degtorad * ((Column - coff) * 2**16 / cfac)  # 열 좌표
    y = degtorad * ((dim_y - Line - 1 - loff) * 2**16 / lfac)  # 행 좌표 (뒤집기 적용)
    # 모든 좌표에 대해 x, y 출력 (디버깅용)
    # print(f"Line={Line}, Column={Column}, x={x}, y={y}")
    # Sd 계산에서 기존 스크립트와 동일한 스케일링 적용
    # Sd = np.sqrt((H * np.cos(x) * np.cos(y))**2 - (np.cos(y)**2 + 1.006739501 * np.sin(y)**2) * (a**2 / (1000 * 1000)))
    Sd = np.sqrt((42164.0 * np.cos(x) * np.cos(y))**2 - (np.cos(y)**2 + 1.006739501 * np.sin(y)**2) * 1737122264)
    if Sd < 0:  # 무효값 방지 및 디버깅
        print(f"Sd negative: {Sd} at (x={x}, y={y}, Line={Line}, Column={Column})")
        return None, None
    Sn = (H * np.cos(x) * np.cos(y) - Sd) / (np.cos(y)**2 + 1.006739501 * np.sin(y)**2)
    S1 = H - (Sn * np.cos(x) * np.cos(y))
    S2 = Sn * (np.sin(x) * np.cos(y))
    S3 = -Sn * np.sin(y)
    Sxy = np.sqrt((S1 * S1) + (S2 * S2))
    nlon = (np.arctan(S2 / S1) + sub_lon) / degtorad
    nlat = np.arctan((1.006739501 * S3) / Sxy) / degtorad
    return nlon, nlat

# 결과 리스트 초기화
result = []
step = 10  # 샘플링 간격 (초기 버전 요청에 따라 10으로 설정)

# 각 픽셀에 대해 위도/경도 변환 및 CTT 데이터 추출
for y in range(0, dim_y, step):
    for x in range(0, dim_x, step):
        ctt_value = ctt_raw[y, x]
        # _FillValue(65535) 제외 및 유효 범위 확인
        if ctt_value != 65535 and 0 <= ctt_value <= 35000:  # CTT의 _FillValue와 유효 범위
            lon, lat = latlon_from_geos(y, x, sub_lon, coff, cfac, loff, lfac)
            if lon is not None and lat is not None:
                if -80 <= lat <= 80 and -180 <= lon <= 190:  # 남/북 범위 확장
                    # CTT 값 스케일링 적용 (K 단위)
                    # ctt_adjusted = ctt_value * 0.01  # scale_factor = 0.01, add_offset = 0.0
                    ctt_adjusted = ctt_value # scale_factor = 0.01, add_offset = 0.0
                    result.append([
                        float(lon),  # 경도
                        float(lat),  # 위도
                        float(ctt_adjusted)  # CTT 값 (K 단위)
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

# JSON 파일로 저장
output_filename = f"output_ctps_fd_ge_step{step}_CTT.json"
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

    scatter = ax.scatter(lons, lats, c=ctt_values, cmap='viridis', s=1, transform=ccrs.PlateCarree())
    plt.colorbar(scatter, label='Cloud Top Temperature (K)')
    plt.title('Geostationary to WGS84 Converted Data (CTT from GK-2A Full Disk)')
    plt.show()

# 파일 닫기
ds.close()