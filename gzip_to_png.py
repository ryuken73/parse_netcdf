from pathlib import Path
from PIL import Image
import gzip
import json
import numpy as np

def json_to_png(json, outfile, image_size=(600, 520), bounds=[76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447]):
  print(outfile)
  """
  주어진 [[lon, lat, value], ...] 데이터를 600x520 이미지로 변환.
  griddata 없이 직접 매핑.
  """
  # 경계 설정
  lon_min, lat_min, lon_max, lat_max = bounds
  
  # 이미지 크기
  width, height = image_size  # 600x520
  
  # 위경도 간격 계산
  lon_step = (lon_max - lon_min) / (width - 1)
  lat_step = (lat_max - lat_min) / (height - 1)
  
  # 2D 배열 초기화 (값 저장용)
  grid_values = np.full((height, width), -9999, dtype=np.float32)
  
  # 데이터를 2D 배열에 매핑
  for lon, lat, value in json:
      # 위경도를 픽셀 인덱스로 변환
      col = int((lon - lon_min) / lon_step)  # 경도 -> 열 인덱스
      row = int((lat_max - lat) / lat_step)  # 위도 -> 행 인덱스 (위도가 상단에서 하단으로 감소)
      
      # 인덱스가 범위 내에 있는지 확인
      if 0 <= row < height and 0 <= col < width:
          grid_values[row, col] = value
  
  # 색상 매핑 (검정-흰색 보간)
  def get_color_from_value(value):
      if value == -9999:
          return [0, 0, 0, 0]  # 투명
      t_min, t_max = -100, 30
      ratio = min(max((value - t_min) / (t_max - t_min), 0), 1)
      r = int(255 * (1 - ratio))
      g = int(255 * (1 - ratio))
      b = int(255 * (1 - ratio))
      return [r, g, b, 255]  # RGBA
  
  # 이미지 데이터 생성 (RGBA)
  image_data = np.zeros((height, width, 4), dtype=np.uint8)
  for i in range(height):
      for j in range(width):
          image_data[i, j] = get_color_from_value(grid_values[i, j])
  
  # 이미지를 PNG로 저장
  image = Image.fromarray(image_data, 'RGBA')
  image.save(outfile)
  print(f'save done: {outfile}') 
  return bounds
  


directory = Path('./jsonfiles')
if __name__ == '__main__' :
  json_files = [f.name for f in directory.glob("*step4.json.gz") if f.is_file()]   
  for json_file_name in json_files:
    full_file_name = directory.joinpath(json_file_name)
    out_file_name = directory.joinpath(f'{Path(json_file_name).stem}.png')
    print(f'gzip to png : {full_file_name}')
    with gzip.open(full_file_name) as json_data:
      json_to_png(json.load(json_data), out_file_name)

