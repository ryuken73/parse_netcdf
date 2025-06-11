import netCDF4 as nc
import numpy as np
import json
from math import radians, degrees, sin, cos, tan, asin, atan, atan2, sqrt

# 정지궤도 투영에서 위경도로 변환하는 함수
def geos_to_lonlat(x, y, sub_lon, H, a, b, cfac, lfac, coff, loff):
    try:
        # 이미지 좌표를 스캔 각도로 변환
        x_scan = (x - coff) / cfac * 2**16
        y_scan = (y - loff) / lfac * 2**16
        
        # 라디안 단위로 변환
        x_rad = radians(x_scan)
        y_rad = radians(y_scan)
        
        # 중간 계산
        sd = sin(x_rad)**2 + cos(x_rad)**2 * cos(y_rad)**2
        discriminant = H**2 * cos(x_rad)**2 * cos(y_rad)**2 - sd * (H**2 - a**2)
        
        # 제곱근 내부 값이 음수인지 확인
        if discriminant < 0:
            return None, None  # 유효하지 않은 좌표
        
        sn = (H * cos(x_rad) * cos(y_rad) - sqrt(discriminant)) / sd
        s1 = H - sn * cos(x_rad) * cos(y_rad)
        s2 = sn * sin(x_rad) * cos(y_rad)
        s3 = -sn * sin(y_rad)
        sxy = sqrt(s1**2 + s2**2)
        
        # 위경도 계산
        lon = sub_lon + degrees(atan2(s2, s1))
        lat = degrees(atan((b**2 / a**2) * s3 / sxy))
        
        # 유효 범위 체크
        # if lon < -179 or lon > 180 or lat < -90 or lat > 90:
        #     return None, None
        return lon, lat
    except Exception as e:
        print(f"Error at (x={x}, y={y}): {e}")
        return None, None

# .nc 파일 경로
file_path = r"D:\002.Code\001.python\netcdf\ir105_fd_ge_202502170000.nc"
step = 1000  # 샘플링 간격

# .nc 파일 열기
dataset = nc.Dataset(file_path, format='NETCDF4')

# 이미지 데이터 읽기
image_pixel_values = dataset.variables['image_pixel_values'][:]

# 투영 파라미터 읽기
sub_lon = dataset.getncattr('sub_longitude')
H = dataset.getncattr('nominal_satellite_height')
a = dataset.getncattr('earth_equatorial_radius')
b = dataset.getncattr('earth_polar_radius')
cfac = dataset.getncattr('cfac')
lfac = dataset.getncattr('lfac')
coff = dataset.getncattr('coff')
loff = dataset.getncattr('loff')

# 결과 리스트
result = []

# 이미지 크기
ny, nx = image_pixel_values.shape

# 이미지 좌표를 위경도로 변환하며 데이터 추출
for y in range(0, ny, step):
    for x in range(0, nx, step):
        lon, lat = geos_to_lonlat(x, y, sub_lon, H, a, b, cfac, lfac, coff, loff)
        if lon is not None and lat is not None:  # 유효한 좌표만 추가
            pixel_value = float(image_pixel_values[y, x])
            result.append([lon, lat, pixel_value])

# 위경도 범위 출력 (디버깅용)
lons = [point[0] for point in result]
lats = [point[1] for point in result]
print("Longitude 범위:", min(lons) if lons else "No valid points", "to", max(lons) if lons else "No valid points")
print("Latitude 범위:", min(lats) if lats else "No valid points", "to", max(lats) if lats else "No valid points")

# 결과 확인 (처음 5개만 출력)
print("샘플 데이터:", result[:5])

# JSON 파일로 저장
output_file = f"output_{step}_fd3.json"
with open(output_file, 'w') as f:
    json.dump(result, f)

print(f"Data saved to {output_file} with step {step}. Total points: {len(result)}")
dataset.close()