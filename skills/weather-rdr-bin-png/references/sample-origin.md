# 샘플 원본

## 포함된 샘플 파일

이 skill은 아래 샘플 asset을 포함한다.

```text
assets/samples/RDR_CMP_HSP_PUB_202506131900.bin
assets/samples/RDR_to_image_last.py
```

샘플 binary는 아래 파일에서 복사했다.

```text
working_script_samples/RDR_CMP_HSP_PUB_202506131900.bin
```

예상 샘플 크기:

```text
13282434 bytes
```

이 파일은 parser sanity check에 사용한다. 압축 해제된 전체 `.bin`이므로 `read_RDR_bin`이나 `scripts/inspect_rdr_bin.py`에 바로 넣을 수 있다.

## 운영 코드와의 관계

`working_script_samples/RDR_to_image_last.py`는 현재 운영 RDR parser/reprojection 로직의 원형이 된 탐색 스크립트다. 운영 반영 위치:

```text
main_with_watcher_mk_image_RDR.py
parseWithVectorNC.py
```

아래 값은 운영 코드와 일치한다.

- `nx = 2305`
- `ny = 2881`
- `dtype = np.int16`
- `offset = 1024`
- `rain_rate <= -30000` null mask
- `rain_rate /= 100`
- rain `ListedColormap` RGBA table
- rain `BoundaryNorm` boundaries
- LCC source CRS
- `source_center_x = 1121`
- `source_center_y = 1681`
- `source_resolution = 500`
- RGBA channel-by-channel `rasterio.warp.reproject`
- `Resampling.nearest`

샘플 스크립트에는 Matplotlib 시각화와 Mapbox bounds 출력도 포함되어 있다. 이 부분은 탐색/디버그용이며 운영 watcher 흐름에는 포함하지 않는다.

## 빠른 확인

`netcdf` 프로젝트 root에서 실행한다.

```bash
python skills/weather-rdr-bin-png/scripts/inspect_rdr_bin.py skills/weather-rdr-bin-png/assets/samples/RDR_CMP_HSP_PUB_202506131900.bin
```

기대 결과:

```text
size_ok: True
```

미래 parser 변경 후 이 샘플을 읽지 못하면, 기상청 RDR HSP 포맷 변경이 확인되고 샘플이 의도적으로 오래된 데이터가 된 경우가 아니라면 regression으로 본다.
