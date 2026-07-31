# 산출물 계약과 검증

## 예상 산출물

입력 파일 예:

```text
{WATCH_PATH_RDR}/{date}/RDR_CMP_HSP_PUB_YYYYMMDDHHMM.bin
```

예상 출력 디렉터리:

```text
{OUT_PATH_RDR}/{date}/
```

예상 출력 파일:

```text
RDR_CMP_HSP_PUB_YYYYMMDDHHMM_step1.png
RDR_CMP_HSP_PUB_YYYYMMDDHHMM_step1_equi.png
RDR_CMP_HSP_PUB_YYYYMMDDHHMM_step1_equi_normal.png
RDR_CMP_HSP_PUB_YYYYMMDDHHMM_step5.png
RDR_CMP_HSP_PUB_YYYYMMDDHHMM_step10.png
```

Resize target:

```text
step5 = 1703 x 1956
step10 = 1277 x 1467
```

## 최소 검증

코드 수정 후:

```bash
python -m py_compile main_with_watcher_mk_image_RDR.py parseWithVectorNC.py to_epsg3857_keep_size.py
```

실제 RDR `.bin` 파일 기준:

1. `scripts/inspect_rdr_bin.py`로 파일 크기를 확인한다.
2. `callback(path)` 또는 별도 파일 1개 처리 wrapper로 처리한다.
3. 예상 PNG가 모두 존재하고 파일 크기가 0보다 큰지 확인한다.
4. `step1`과 `step1_equi`를 눈으로 확인한다. 완전히 투명하거나 위치가 틀어지면 실패로 본다.
5. normal map이 equirectangular color PNG에서 생성됐는지 확인한다.

## 보조 스크립트 사용법

프로젝트 root에서 실행한다.

```bash
python skills/weather-rdr-bin-png/scripts/inspect_rdr_bin.py /data/node_project/weather_data/in_data/rdr/2025-09-26/RDR_CMP_HSP_PUB_202509260845.bin --out-root /data/node_project/weather_data/out_data/rdr
```

다른 스크립트나 agent가 결과를 파싱해야 하면 `--json`을 사용한다.

## 흔한 증상

`ValueError: cannot reshape array`

- 실제 파일 크기가 `13282434` bytes인지 확인한다.
- dimension을 바꾸기 전에 partial download나 upstream format 변경을 의심한다.

`_step1_equi.png` 누락

- GDAL command가 PATH에 있는지 확인한다.
- `to_epsg3857_keep_size.py`가 `error to convert file`을 출력했는지 확인한다.
- 임시 파일 권한과 출력 디렉터리 권한을 확인한다.

이미지가 완전히 투명함

- null mask threshold를 확인한다.
- 입력이 metadata/error 파일이 아니라 실제 RDR HSP binary인지 확인한다.
- colormap bad-value 처리와 rain-rate scale을 확인한다.

이미지가 공간적으로 밀림

- `reproject_RDR`의 source LCC 상수를 확인한다.
- `to_epsg3857_keep_size.py`의 `wgs84_values['rdr']` bounds를 확인한다.
- source transform이 틀렸는데 최종 GDAL bounds만 조정하지 않는다.

색상이 흐리거나 섞임

- RGBA 채널 재투영에 `Resampling.nearest`가 유지되는지 확인한다.
- 범주형 컬러 이미지를 저장하기 전에 resize하지 않는다.

watcher가 새 파일을 반복적으로 놓침

- `watchFolder_Thread.py`의 파일 크기 안정화 설정을 확인한다.
- 고정 sleep보다 downloader atomic rename을 우선 검토한다.
- PM2 app `watcher_image_rdr`가 실행 중이고 `.env.prod`를 사용 중인지 확인한다.
