import netCDF4 as nc
import numpy as np

import matplotlib.pyplot as plt

nc_file = 'D:/002.Code/001.python/netcdf/working_script_samples/gk2a_ami_le1b_ir105_fd020ge_202502170000.nc'
attr_to_get = 'image_pixel_values'

ds = nc.Dataset(nc_file, format='NETCDF4')  
image_pixel_values = ds.variables[attr_to_get][:]
dim_y, dim_x = image_pixel_values.shape

# image_pixel_values.shape == 5500 X 5500
print("image_pixel_values shape:", image_pixel_values.shape)

# 제공된 NetCDF 파일에서 위경도 테이블 로드
LAT_LON_NC_FILE = '../assets/gk2a_ami_fd020ge_latlon.nc'
latlon_ds = nc.Dataset(LAT_LON_NC_FILE, format='NETCDF4')
lat = latlon_ds.variables['lat'][:]
lon = latlon_ds.variables['lon'][:]
latlon_ds.close()
# lat.shape, lon.shape == 5500 X 5500
print('sample lat lon values:', lat[-1000:], lon[-1000:])
print("latlon_ds shape:", lat.shape, lon.shape)
plt.imshow(lat, cmap='viridis', interpolation='none')
plt.colorbar(label='Latitude')
plt.show()

plt.imshow(lon, cmap='viridis', interpolation='none')
plt.colorbar(label='Longitude')
plt.show()

plt.imshow(image_pixel_values, cmap='viridis', interpolation='none')
plt.colorbar(label='image_pixel_values')
plt.show()

step = 1 
y_indices = np.arange(0, dim_y, step)
x_indices = np.arange(0, dim_x, step)
print("y_indices:", y_indices.shape)
print("x_indices:", x_indices.shape)
Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
print("Y shape:", Y.shape)
print("X shape:", X.shape)
print("Y sample:", Y[-5:, -5:])
print("X sample:", X[-5:, -5:])

# 인덱스를 사용해 위경도 배열 샘플링
sampled_lat = lat[Y, X]
sampled_lon = lon[Y, X]
# filter -999 value from sampled_lat and sampled_lon)
valid_mask = (sampled_lat != -999.0) & (sampled_lon != -999.0)
sampled_lat = sampled_lat[valid_mask]
sampled_lon = sampled_lon[valid_mask]

print("max sampled_lon:", np.max(sampled_lon), "min sampled_lon:", np.min(sampled_lon))
print("max sampled_lat:", np.max(sampled_lat), "min sampled_lat:", np.min(sampled_lat))
print("sampled_lon shape:", sampled_lon.shape)