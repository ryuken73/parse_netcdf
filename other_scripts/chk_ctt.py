import netCDF4 as nc
import numpy as np

# NetCDF 파일 열기
file_path = r"D:\002.Code\099.study\deck.gl.test\src\bigfiles\ctps_ea_lc_202502170000.nc"
ds = nc.Dataset(file_path, "r")

# CTT 데이터 읽기 (원래 ushort 값)
ctt_values = ds.variables["CTT"][:]
fill_value = ds.variables["CTT"]._FillValue  # 65535

# _FillValue 제외한 유효 데이터만 필터링
valid_ctt = ctt_values[ctt_values != fill_value]

# 최소, 최대 값 출력
print("CTT 유효 최소값:", valid_ctt.min(), "CTT 유효 최대값:", valid_ctt.max())
print("CTT 값 분포 (처음 10개 샘플):", valid_ctt[:10])

# step=3으로 샘플링한 데이터 확인
sampled_values = ctt_values[::3, ::3]  # 3칸 간격으로 샘플링
valid_sampled = sampled_values[sampled_values != fill_value]
print("샘플링된 CTT 유효 최소값:", valid_sampled.min(), "샘플링된 CTT 유효 최대값:", valid_sampled.max())
print("샘플링된 CTT 값 분포 (처음 10개 샘플):", valid_sampled[:10])

ds.close()