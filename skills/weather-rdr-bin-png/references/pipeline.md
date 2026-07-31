# RDR 처리 파이프라인

## 진입점

`main_with_watcher_mk_image_RDR.py`는 아래 호출로 시작한다.

```python
start_watching(config.WATCH_PATH_RDR, None, callback)
```

watcher는 `WATCH_PATH_RDR`로 들어오는 기상청 RDR binary 파일을 감시한다. 파일이 준비되면 `callback(rdr_file)`이 실행된다.

출력 경로는 입력 파일의 parent directory 이름을 사용해 만든다.

```text
{OUT_PATH_RDR}/{입력_파일_parent_directory_name}/
```

입력 parent directory는 보통 날짜 폴더다.

## 파일 1개 처리 순서

명확한 운영상 이유가 없다면 이 순서를 유지한다.

1. 출력 날짜 디렉터리를 만든다.
2. `read_RDR_bin(rdr_file)`로 RDR HSP binary를 읽는다.
3. `reproject_RDR(colored_array)`로 RGBA 격자를 재투영한다.
4. 고품질 기본 PNG를 `{stem}_step1.png`로 저장한다.
5. `convert_to_equi_rectangle('rdr', step1, step1_equi)`로 geographic/equirectangular PNG를 만든다.
6. `create_normal_map_for_rdr(step1_equi, output_normal_path=step1_equi_normal)`로 normal map을 만든다.
7. `resize_image`로 step1 이미지를 step5, step10 PNG로 축소한다.

## 런타임 설정

관련 `.env.prod` 값:

```text
OUT_PATH_RDR=/data/node_project/weather_data/out_data/rdr
WATCH_PATH_RDR=/data/node_project/weather_data/in_data/rdr
OS_SEP='/'
```

`config.py`는 이 값들을 `config.OUT_PATH_RDR`, `config.WATCH_PATH_RDR`, `config.OS_SEP`로 노출한다.

## PM2 운영

`ecosystem.config.js`의 PM2 app 이름은 `watcher_image_rdr`이다.

기대 설정:

- script: `./main_with_watcher_mk_image_RDR.py`
- interpreter: `python3`
- instances: `1`
- exec_mode: `fork`
- env_file: `.env.prod`
- out log: `./logs/watcher-image-rdr-out.log`
- error log: `./logs/watcher-image-rdr-error.log`

## watcher 완료 판정

`watchFolder_Thread.py`는 파일 생성 이벤트를 받은 뒤 파일 크기가 안정될 때까지 확인하고 callback을 호출한다.

미완성 파일을 읽는 문제가 의심되면 fetcher/downloader 쪽을 우선 본다. 가장 안정적인 방식은 임시 확장자나 임시 파일명으로 다운로드한 뒤 완료 시 `.bin` 파일명으로 atomic rename하는 것이다. 고정 sleep을 늘리는 방식은 빈도를 낮출 수는 있지만 완료를 보장하지 않는다.

## 누락 파일 재처리 설계

누락된 RDR 파일을 재처리할 때도 watcher의 파일 1개 처리 로직과 같은 흐름을 사용한다.

batch worker를 만들 때는 다음 규칙을 지킨다.

- 입력 날짜 root 아래 `.bin` 파일을 재귀적으로 찾는다.
- 출력 날짜 디렉터리는 watcher와 동일하게 입력 parent directory에서 얻는다.
- 모든 예상 산출물이 존재하고 파일 크기가 0보다 클 때만 skip한다.
- Rasterio/GDAL과 RGBA 배열은 메모리를 많이 쓰므로 병렬 worker 수를 제한한다.
- native library 상태 격리가 필요하면 파일 1개를 subprocess에서 처리한다.

batch 로직을 만들거나 실행하기 전에 `scripts/inspect_rdr_bin.py`로 후보 `.bin`의 크기와 예상 산출물 경로를 확인한다.
