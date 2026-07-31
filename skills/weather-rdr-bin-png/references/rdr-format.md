# RDR 포맷과 투영

## Binary layout

`read_RDR_bin`은 현재 기상청 RDR HSP binary layout을 아래처럼 가정한다.

```text
header_offset_bytes = 1024
dtype = np.int16
rdr_nx = 2305
rdr_ny = 2881
shape = (rdr_ny, rdr_nx)
expected_file_size = 1024 + 2305 * 2881 * 2 = 13282434 bytes
```

파싱 순서:

1. 파일 bytes 전체를 읽는다.
2. offset `1024`부터 signed 16-bit integer로 해석한다.
3. `float32`로 변환한다.
4. `(2881, 2305)`로 reshape한다.
5. `<= -30000` 값을 null로 처리한다.
6. 강수량을 `/ 100`으로 scale한다.

reshape 오류가 나면 먼저 입력 파일 크기를 확인한다. 파일 크기가 맞지 않으면 partial download, 잘못된 파일 종류, upstream format 변경을 의심한다.

## 강수 컬러 boundary

강수량은 `BoundaryNorm`과 `ListedColormap`으로 색상에 매핑한다. null 영역은 `colormap_rain.set_bad([0, 0, 0, 0])`로 투명 처리한다.

현재 boundary:

```python
[0, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 90, 110, 150]
```

현재 RGBA 컬러 테이블:

```python
[
    [0, 200, 255, 0], [0, 155, 245, 255], [0, 74, 245, 255],
    [0, 255, 0, 255], [0, 190, 0, 255], [0, 140, 0, 255], [0, 90, 0, 255],
    [255, 255, 0, 255], [255, 220, 31, 255], [249, 205, 0, 255],
    [224, 185, 0, 255], [204, 170, 0, 255],
    [255, 102, 0, 255], [255, 50, 0, 255], [210, 0, 0, 255], [180, 0, 0, 255],
    [224, 169, 255, 255], [201, 105, 255, 255], [179, 41, 255, 255], [147, 0, 228, 255],
    [179, 180, 222, 255], [76, 78, 177, 255], [0, 3, 144, 255],
    [51, 51, 51, 255], [51, 51, 51, 255],
]
```

`read_RDR_bin`의 출력은 반드시 `uint8` RGBA여야 한다.

## 재투영

`reproject_RDR`는 원본 격자를 Lambert Conformal Conic으로 보고 EPSG:3857로 변환한다.

source 상수:

```text
source_width = 2305
source_height = 2881
source_center_x = 1121
source_center_y = 1681
source_resolution = 500
source_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
```

source bounds:

```python
{
    "left": -1121 * 500,
    "bottom": (2881 - 1681) * 500,
    "right": (2305 - 1121) * 500,
    "top": -1681 * 500,
}
```

RGBA 4개 채널은 각각 `rasterio.warp.reproject`로 변환한다. 강수 등급 색상이 섞이지 않도록 `Resampling.nearest`를 유지한다.

## Equirectangular 변환 범위

`to_epsg3857_keep_size.py`의 `wgs84_values['rdr']` 설정:

```python
{
    "UL": (118.8394260710767, 43.572496647155695),
    "LR": (133.5627133041138, 30.102047565010807),
    "width": 2554,
    "height": 2934,
}
```

변환 순서:

1. `gdal_translate`로 중간 PNG에 EPSG:3857 georeference를 부여한다.
2. `gdalwarp`로 EPSG:4326으로 변환한다.
3. RDR `wgs84_values`의 target extent와 target size를 유지한다.

운영 환경에서는 `gdal_translate`, `gdalwarp`가 PATH에 있어야 한다.

## Normal map

`create_normal_map_for_rdr`는 equirectangular color PNG를 읽고, 정확히 일치하는 RGB 값을 높이값으로 매핑한 뒤 gradient로 normal map을 만든다.

normal-map 컬러 테이블은 24개 RGB row이고, `read_RDR_bin`의 강수 컬러맵은 25개 RGBA row다. 강수 컬러맵의 마지막 dark gray duplicate는 같은 default/dark class로 취급되어 normal-map 테이블에 별도 row로 들어가지 않는다.

현재 intensity table:

```python
[0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 24, 30, 31, 32, 33, 40, 41, 42, 43, 60, 61, 62]
```

강수 컬러를 바꾸면 `read_RDR_bin`과 `create_normal_map_for_rdr`를 함께 검토한다.
