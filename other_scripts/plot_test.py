import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Lambert Conformal 투영 설정
projection = ccrs.LambertConformal(central_longitude=126, central_latitude=38, standard_parallels=(30, 60))

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})

# 해안선 및 국가 경계선 추가
ax.coastlines(resolution='10m', linewidth=1.0)  # 해안선
ax.add_feature(cfeature.BORDERS, linewidth=1.0)  # 국가 경계선
ax.add_feature(cfeature.LAND, facecolor="lightgray")  # 육지 색상 추가

plt.show()
