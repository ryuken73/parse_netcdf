import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
from pyproj import Transformer, CRS
# from mk_image_mercator_lcc_interpo_color import generate_image_from_data
from mk_image_mercator_geo_color import generate_image_from_data

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
dim_y, dim_x = image_pixel_values.shape
print("image_pixel_values shape:", image_pixel_values.shape)

sub_lon = ds.getncattr('sub_longitude')
H = ds.getncattr('nominal_satellite_height')
a = ds.getncattr('earth_equatorial_radius')
b = ds.getncattr('earth_polar_radius')
cfac = ds.getncattr('cfac')
lfac = ds.getncattr('lfac')
coff = ds.getncattr('coff')
loff = ds.getncattr('loff')

y_indices = np.arange(0, dim_y, step)
x_indices = np.arange(0, dim_x, step)
Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
degtorad = np.pi / 180.0
x = degtorad * ((X - coff) * 2**16 / cfac)
y = degtorad * ((dim_y - Y - 1 - loff) * 2**16 / lfac)
cos_x = np.cos(x)
cos_y = np.cos(y)
sin_y = np.sin(y)
Sd = np.sqrt((42164.0 * cos_x * cos_y)**2 - (cos_y**2 + 1.006739501 * sin_y**2) * 1737122264)
valid_mask = Sd >= 0
Sn = np.where(valid_mask, (42164.0 * cos_x * cos_y - Sd) / (cos_y**2 + 1.006739501 * sin_y**2), np.nan)
S1 = 42164.0 - (Sn * cos_x * cos_y)
S2 = Sn * (np.sin(x) * cos_y)
S3 = -Sn * sin_y
Sxy = np.sqrt(S1**2 + S2**2)
lon = np.where(valid_mask, (np.arctan2(S2, S1) + sub_lon) / degtorad, np.nan)
lat = np.where(valid_mask, np.arctan2(1.006739501 * S3, Sxy) / degtorad, np.nan)
# latlon_mask = (lat >= -80) & (lat <= 80) & (lon >= -180) & (lon <= 190)
latlon_mask = (lat >= -80) & (lat <= 80) & (lon >= 60) & (lon <= 180)
final_mask = valid_mask & latlon_mask

# 변환 테이블 적용
values = image_pixel_values[Y, X] 
print('type of values', values)
conversion_file = "ir105_conversion_c.txt"
lut = load_conversion_table(conversion_file)

# 마스크를 사용한 안전한 변환
mask = (values >= 0) & (values < len(lut))
converted_values = np.full_like(values, -9999, dtype=float)  # 기본값 -9999
converted_values[mask] = lut[values[mask]]  # 유효한 값만 변환

print("Max value in converted_values:", np.max(converted_values))
print("Min value in converted_values:", np.min(converted_values))

# result에 converted_values 사용
result = np.column_stack([
    lon[final_mask].astype(float),
    lat[final_mask].astype(float),
    converted_values[final_mask].astype(float)
])

# 디버깅 출력
lons = result[:, 0]
lats = result[:, 1]
print("Longitude 범위:", np.min(lons) if lons.size > 0 else "No valid points", 
      "to", np.max(lons) if lons.size > 0 else "No valid points")
print("Latitude 범위:", np.min(lats) if lats.size > 0 else "No valid points", 
      "to", np.max(lats) if lats.size > 0 else "No valid points")
print("샘플 데이터:", result[:5])

data_list = result.tolist()

# PNG 생성
output_path = "output_image_fast.png"
bounds = [60, -80, 180, 80]
# generate_image_from_data(result, output_path, image_size=(2048, 2048), bounds=bounds)
# generate_image_from_data(data_list, output_path, image_size=(2048, 2048), bounds=bounds)
# generate_image_from_data(data_list, output_path, image_size=(1200, 1200), bounds=bounds)
generate_image_from_data(data_list, output_path, image_size=(3192, 3192), bounds=bounds)

ds.close()

# Matplotlib과 Cartopy를 사용한 시각화
# if lons.size > 0 and lats.size > 0:
#     plt.figure(figsize=(10, 8))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.set_extent([-180, 190, -80, 80], crs=ccrs.PlateCarree())  # 남/북 범위 확장
#     ax.add_feature(cfeature.COASTLINE)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
#     ax.add_feature(cfeature.OCEAN, facecolor='lightgray')
#     ax.add_feature(cfeature.LAND, facecolor='lightgreen')

#     scatter = ax.scatter(lons, lats, c=result[:, 2], cmap='viridis', s=1, transform=ccrs.PlateCarree())
#     plt.colorbar(scatter, label='Pixel Value')
#     plt.title('Geostationary to WGS84 Converted Data (GK-2A Full Disk)')
#     plt.show()

# # 파일 닫기
# ds.close()