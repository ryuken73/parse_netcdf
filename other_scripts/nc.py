import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# NetCDF 파일 열기
a_ds = xr.open_dataset("ir105.nc")  # 적외선 온도
b_ds = xr.open_dataset("ctps.nc")  # 구름 온도

# 온도 데이터 변수명 확인 후 로드
a_var = list(a_ds.data_vars.keys())[0]  # a.nc에서 첫 번째 변수 선택
b_var = list(b_ds.data_vars.keys())[0]  # b.nc에서 첫 번째 변수 선택

a_data = a_ds[a_var]  # a.nc의 데이터
b_data = b_ds[b_var]  # b.nc의 데이터

# 온도가 20도 이하인 부분은 b_data를, 나머지는 a_data를 사용
combined_data = xr.where(a_data <= 20, b_data, a_data)

# 시각화
plt.figure(figsize=(10, 6))
plt.pcolormesh(combined_data, cmap="coolwarm")  # 색상 맵 설정
plt.colorbar(label="Temperature (°C)")
plt.title("Merged Temperature Data (A & B)")
plt.show()
