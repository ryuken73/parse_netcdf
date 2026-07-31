---
name: weather-rdr-bin-png
description: 기상청 Radar/RDR HSP binary 파일을 PNG 이미지로 변환하는 netcdf 프로젝트 파이프라인을 구현, 유지보수, 진단할 때 사용한다. main_with_watcher_mk_image_RDR.py, KMA radar .bin 수신/파싱, read_RDR_bin, reproject_RDR, RDR PNG/equi/normal-map 산출물, WATCH_PATH_RDR/OUT_PATH_RDR watcher 동작, 누락 파일 재처리, RDR binary/output 검증 작업에 사용한다.
---

# Weather RDR Bin PNG

## 빠른 시작

`netcdf` 프로젝트에서 기상청 RDR binary를 PNG로 만드는 작업을 할 때 이 skill을 사용한다. 먼저 실제 repo 파일을 확인하고, 필요한 경우에만 아래 참조 문서를 읽는다.

핵심 repo 파일:

- `main_with_watcher_mk_image_RDR.py`: watcher 진입점과 파일 1개 처리 순서
- `parseWithVectorNC.py`: `read_RDR_bin`, `reproject_RDR`, `create_normal_map_for_rdr`, `resize_image`
- `to_epsg3857_keep_size.py`: `convert_to_equi_rectangle('rdr', ...)`
- `config.py`, `.env.prod`: `WATCH_PATH_RDR`, `OUT_PATH_RDR`, `OS_SEP`
- `watchFolder_Thread.py`: 파일 생성 감지와 쓰기 완료 대기

## 참조 문서

- watcher 흐름, 파일 1개 처리, batch/backfill 설계, PM2 운영, 실패 처리를 바꿀 때는 [references/pipeline.md](references/pipeline.md)를 읽는다.
- binary 파싱, 강수 컬러 매핑, 투영 상수, GDAL/equi 변환 범위, normal map 높이 규칙을 바꿀 때는 [references/rdr-format.md](references/rdr-format.md)를 읽는다.
- 예상 산출물 이름, 검증 절차, 장애 증상, 운영 로그를 확인할 때는 [references/output-validation.md](references/output-validation.md)를 읽는다.
- 운영 코드와 원형 테스트 스크립트 `working_script_samples/RDR_to_image_last.py`를 비교할 때는 [references/sample-origin.md](references/sample-origin.md)를 읽는다.

## 보조 스크립트

[scripts/inspect_rdr_bin.py](scripts/inspect_rdr_bin.py)는 프로젝트 모듈을 import하지 않고 RDR `.bin` 파일을 점검한다.

```bash
python skills/weather-rdr-bin-png/scripts/inspect_rdr_bin.py /path/to/RDR_CMP_HSP_PUB_202509260845.bin --out-root /data/node_project/weather_data/out_data/rdr
```

이 스크립트는 예상 binary byte 크기를 확인하고, `--out-root`가 있으면 예상 PNG 산출물과 누락 여부를 출력한다. 이미지를 생성하지는 않는다.

skill에 포함된 샘플 binary를 점검하려면:

```bash
python skills/weather-rdr-bin-png/scripts/inspect_rdr_bin.py skills/weather-rdr-bin-png/assets/samples/RDR_CMP_HSP_PUB_202506131900.bin
```

## 작업 규칙

- 기상청 원천 RDR 포맷이 바뀐 것이 확인되지 않았다면 기존 dimension, offset, CRS, color boundary, output filename 규칙을 유지한다.
- 강수 레벨 이미지는 범주형 색상이므로 재투영 시 nearest-neighbor resampling을 유지한다. bilinear/cubic으로 바꾸지 않는다.
- 병렬 처리를 추가할 때는 worker 수를 반드시 제한한다. RDR 처리는 NumPy/Rasterio/GDAL 메모리 사용량이 크다.
- 수신 중인 `.bin`을 읽는 문제가 의심되면 sleep 추가보다 임시 파일명 다운로드 후 atomic rename 구조를 우선 검토한다.
- 코드 수정 후 최소한 `python -m py_compile main_with_watcher_mk_image_RDR.py parseWithVectorNC.py to_epsg3857_keep_size.py`를 실행한다.

## 포함 자산

- `assets/samples/RDR_CMP_HSP_PUB_202506131900.bin`: 정상으로 확인된 기상청 RDR HSP 샘플 binary, 13,282,434 bytes
- `assets/samples/RDR_to_image_last.py`: 운영 함수들의 원형이 된 탐색/테스트 스크립트
