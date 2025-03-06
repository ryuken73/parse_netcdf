import xarray as xr
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def getproj (proj_type) :
  if(proj_type == 'lambert_conformal_conic') :
    return 'lcc'
  return ''

def projection_to_coords (ds, type):
  dsObj = ds if type == None else ds['gk2a_imager_projection']
  proj_type = ds.attrs.get('projection_type') if type == None else dsObj.attrs.get('grid_mapping_name')
  lat_1 = dsObj.attrs.get('standard_parallel1')
  lat_2 = dsObj.attrs.get('standard_parallel2')
  lat_0 = dsObj.attrs.get('origin_latitude')
  lon_0 = dsObj.attrs.get('central_meridian')
  print(lat_1, lat_2, lat_0, lon_0)
  proj = getproj(proj_type)
  # print(proj_type, proj)
  proj_lcc = pyproj.Proj(proj=proj,
                          lat_1=lat_1, lat_2=lat_2,  # 표준 위도
                          lat_0=lat_0, lon_0=lon_0,  # 원점 위도/경도
                          x_0=0, y_0=0, datum="WGS84")

  # X, Y 좌표 만들기
  x = np.arange(ds.dims["dim_x"])  # 픽셀 인덱스
  y = np.arange(ds.dims["dim_y"])  # 픽셀 인덱스

  # 투영 좌표를 위경도로 변환
  lon, lat = np.meshgrid(x, y)
  lon, lat = proj_lcc(lon, lat, inverse=True)

  # 좌표 추가
  ds = ds.assign_coords({"latitude": (("dim_y", "dim_x"), lat),
                        "longitude": (("dim_y", "dim_x"), lon)})
  # 지도표시를 위한 CRS (Coordinate Reference Ssytem) 계산
  map_crs = ccrs.LambertConformal(
    central_longitude=lon_0,
    central_latitude=lat_0,
    standard_parallels=(lat_1, lat_2)
  )
  return ds, map_crs
  # return ds

def geo_to_coords (ds) :
  gk2a_proj = ds['gk2a_imager_projection']
  h = gk2a_proj.attrs.get('perspective_point_height')
  lon_0 = gk2a_proj.attrs.get('longitude_of_projection_origin')
  ma = gk2a_proj.attrs.get('semi_major_axis')
  # print(gk2a_proj, ds.attrs.get('gk2a_imager_projection'), h, lon_0, ma)
  print(gk2a_proj)
  print(gk2a_proj.attrs.get('grid_mapping_name'))
  proj_geo = pyproj.Proj(proj="geos",
                          h=h,  # 위성 고도
                          lon_0=lon_0,   # 원점 경도
                          x_0=0, y_0=0,
                          a=ma, b=6356752.3142,  # 지구 타원체 (WGS84)
                          sweep="x")

  # X, Y 픽셀 좌표 생성
  x = (np.arange(ds.dims["xdim"]) - 1125.5) * 2000  # 2km 해상도
  y = (np.arange(ds.dims["ydim"]) - 2337.5) * 2000

  # 격자로 변환
  lon, lat = np.meshgrid(x, y)
  lon, lat = proj_geo(lon, lat, inverse=True)  # 위경도 변환

  # 좌표 추가
  ds = ds.assign_coords({"latitude": (("ydim", "xdim"), lat),
                        "longitude": (("ydim", "xdim"), lon)})

  return ds


# NetCDF 파일 열기
ir105_ea = xr.open_dataset("ir105_ea_lc_202502170000.nc")
ctps_ea = xr.open_dataset("ctps_ea_lc_202502170000.nc")

a_ds, map_crs = projection_to_coords(ir105_ea, None)
b_ds, map_crs1 = projection_to_coords(ctps_ea, 'CTPS')

# 파일의 전체 구조 확인
# print(a_ds)
# print(ctps_ea)



# 좌표 변수 확인
# print("A.nc 좌표:", list(a_ds.coords))
# print("B.nc 좌표:", list(b_ds.coords))

# # # 위도, 경도 값 비교
print("A.nc 위도 범위:", a_ds.coords.get("latitude", a_ds.coords.get("lat", None)))
# print("B.nc 위도 범위:", b_ds.coords.get("latitude", b_ds.coords.get("lat", None)))

print("A.nc 경도 범위:", a_ds.coords.get("longitude", a_ds.coords.get("lon", None)))
# print("B.nc 경도 범위:", b_ds.coords.get("longitude", b_ds.coords.get("lon", None)))

# 격자 간격 확인 (해상도)
# a_lat = a_ds.coords.get("latitude", a_ds.coords.get("lat"))
# b_lat = b_ds.coords.get("latitude", b_ds.coords.get("lat"))

# a_lon = a_ds.coords.get("longitude", a_ds.coords.get("lon"))
# b_lon = b_ds.coords.get("longitude", b_ds.coords.get("lon"))

# print("A.nc 위도 간격:", a_lat.diff(dim="latitude").values if a_lat is not None else "없음")
# print("B.nc 위도 간격:", b_lat.diff(dim="latitude").values if b_lat is not None else "없음")

# print("A.nc 경도 간격:", a_lon.diff(dim="longitude").values if a_lon is not None else "없음")
# print("B.nc 경도 간격:", b_lon.diff(dim="longitude").values if b_lon is not None else "없음")

print("ir105 차원 목록:", list(a_ds.dims))
print("ir105 변수 목록:", list(a_ds.data_vars))
print("ir105 좌표 목록:", list(a_ds.coords))
print("ctps 차원 목록:", list(b_ds.dims))
print("ctps 변수 목록:", list(b_ds.data_vars))
print("ctps 좌표 목록:", list(b_ds.coords))

ir105_value = a_ds.data_vars['image_pixel_values']
ctt_value = b_ds.data_vars['CTT']


lat = a_ds['latitude']
lon = a_ds['longitude']

print("Longitude range:", lon.min(), lon.max())
print("Latitude range:", lat.min(), lat.max())

print("Longitude sample:", lon[:5])
print("Latitude sample:", lat[:5])

# 그림 설정
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': map_crs})
ax.set_facecolor('white');

# 데이터 표시 (컬러맵 설정)
# img = ax.pcolormesh(lon, lat, ctt_value, cmap='coolwarm', transform=ccrs.PlateCarree(), alpha=0.01, vmin=230, vmax=310, zorder=1)
img = ax.pcolormesh(lon, lat, ir105_value, cmap='coolwarm', transform=ccrs.PlateCarree(), alpha=0.6)

# 지도(해안선) 추가
ax.coastlines(resolution='110m', color='black', linewidth=1, zorder=3)  # 해안선 추가
ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor="blue")  # 국가 경계선 추가 (선택 사항)
ax.add_feature(cfeature.LAND, facecolor='yellow')  # 육지 색상 추가
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue')  # 바다 색상 추가


# 컬러바 추가
# cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.8)
# cbar.set_label("Cloud Temperature (°C)")

# 제목 설정
plt.title("Cloud Temperature with Coastline")
plt.show()

# 시각화
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(ir105_value, cmap="coolwarm")  # 색상 맵 설정
# plt.colorbar(label="Temperature (°C)")
# plt.title("IR105")
# plt.show()