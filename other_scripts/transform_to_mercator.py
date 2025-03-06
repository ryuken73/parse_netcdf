import json
from pyproj import Proj, Transformer
import numpy as np

# 입력 JSON 읽기
input_file = "output_ir105_3.json"
output_file = "output_ir105_3_web_mercator.json"

with open(input_file, "r") as f:
    data = json.load(f)

# WGS84 (EPSG:4326) -> Web Mercator (EPSG:3857) 변환
wgs84 = Proj(proj="latlong", datum="WGS84")
web_mercator = Proj(proj="merc", lat_ts=0, lon_0=0, ellps="WGS84")
transformer = Transformer.from_proj(wgs84, web_mercator)

# 데이터 변환 (lon, lat -> Web Mercator x, y)
result = []
for item in data:
    lon, lat, ctt = item
    try:
        x, y = transformer.transform(lon, lat)
        result.append([x, y, ctt])  # [Web Mercator x, y, CTT]
    except Exception as e:
        print(f"변환 오류: {lon}, {lat} -> {e}")

# 결과 저장
with open(output_file, "w") as f:
    json.dump(result, f)
print(f"Web Mercator 데이터가 {output_file}에 저장되었습니다.")