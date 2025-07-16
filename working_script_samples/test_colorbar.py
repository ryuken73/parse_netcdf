import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

# 샘플 데이터 생성
x = np.linspace(-1, 200, 100)
y = np.linspace(-1, 200, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y) * 5  # -5에서 5 사이의 값

print(X, Y, Z)

# ListedColormap 설정
# colors = ['white', 'yellow', 'blue', 'purple', 'black']
# cmap = ListedColormap(colors)
cmap = ListedColormap(np.array([
    [255, 255, 255, 0], [0, 200, 255, 255], [0, 155, 245, 255], [0, 74, 245, 255],  # 하늘색
    [0, 255, 0, 255], [0, 190, 0, 255], [0, 140, 0, 255], [0, 90, 0, 255],        # 초록색
    [255, 255, 0, 255], [255, 220, 31, 255], [249, 205, 0, 255], [224, 185, 0, 255], [204, 170, 0, 255],  # 노랑색
    [255, 102, 0, 255], [255, 50, 0, 255], [210, 0, 0, 255], [180, 0, 0, 255],    # 빨간색
    [224, 169, 255, 255], [201, 105, 255, 255], [179, 41, 255, 255], [147, 0, 228, 255],  # 보라색
    [179, 180, 222, 255], [76, 78, 177, 255], [0, 3, 144, 255], [51, 51, 51, 255]  # 파란색
]) / 255)

# BoundaryNorm 설정
# levels = [-np.inf, 0, 1, 2, 3, np.inf]  # 경계값
levels = np.array([
    0, 0.1, 0.5, 1,  # 하늘색
    2, 3, 4, 5,      # 초록색
    6, 7, 8, 9, 10,  # 노랑색
    15, 20, 25, 30,  # 빨간색
    40, 50, 60, 70,  # 보라색
    90, 110, 150     # 파란색
])
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

# 플롯 생성
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(X, Y, Z, cmap=cmap, norm=norm, extend='both')

# 컬러바 추가
cbar = fig.colorbar(cf, ax=ax, extend='both')
cbar.set_label('Value')
cbar.set_ticks([0, 1, 2, 3])  # 컬러바에 주요 경계값 표시

# 플롯 제목 및 축 레이블
ax.set_title('BoundaryNorm with ListedColormap')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.tight_layout()
plt.show()