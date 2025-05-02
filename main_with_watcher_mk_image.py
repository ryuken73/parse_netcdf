import os
from pathlib import Path
from parseWithVectorNC import *
from watchFolder_Thread import start_watching
from config import get_config
import sys
import multiprocessing

config = get_config()
print(f'Running in {config.ENV} mode')
print(f'OUT_PATH = {config.OUT_PATH}')
print(f'WATCH_PATH = {config.WATCH_PATH}')

SAVE_IMAGE_STEPS = {
  'ea': [1, 5, 10],
  'fd': [1, 5, 10],
  'ko': [1, 5, 10]
}
IMAGE_SIZE = {
  'ea': {
    # 1: (3000, 2600),
    1: (1500, 1300),
    5: (1200, 1040),
    10: (900, 780),
  },
  'fd': {
    # 1: (5500, 5500),
    1: (3192, 3192),
    5: (2048, 2048),
    10: (1400, 1400),
  },
  'ko': {
    1: (800, 800),
    5: (700, 700),
    10: (600, 600),
  }
}
BOUNDS ={
  'ea': [76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447],
  'fd': [30, -80, 220, 80],
  'ko':  [113.99641, 29.312252, 138.003582, 45.728965]
}

# out_dir = 'D:/002.Code/002.node/weather_api/data/weather/gk2a'
out_dir = config.OUT_PATH
conversion_file = 'ir105_conversion_c.txt'
use_index = 1

## nc 1개에 대해서 아래 파일들 생성
## ea 파일 (json + image)
##  mono png: step1(1500X1300, 1분, 1M), convert_1200X1040, convert_900X780
##  color png: step1(1500X1300, 4분, 2M),convert_1200X1040, convert_900X780
## fd 파일 (image only)
##  gzip: none
##  mono png: step1(3192X3192, 2분, 6M),  convert_2048, convert_1400
##  color png: step1(3192X3192, 5분, 10M),  convert_2048, convert_1400
## ko 파일 (image only)
##  gzip: none
##  mono png: step1(800X800, , 500KB),  convert_700, convert_600
##  color png: step1(800X800, , 600KB),  convert_700, convert_600

def callback(nc_file):
  print(f'processing file {nc_file}')
  sub_dir = os.path.dirname(nc_file).split(config.OS_SEP)[-1:][0]
  print(sub_dir)
  save_dir = f"{out_dir}/{sub_dir}"
  os.makedirs(save_dir, exist_ok=True)
  print(f'save file to {save_dir}')
  nc_coverage = get_nc_coverage(nc_file)
  # for step in save_steps_sorted :
  highest_step = min(SAVE_IMAGE_STEPS[nc_coverage])
  out_file = get_json_fname(save_dir, nc_file, highest_step)
  attr_to_get = 'image_pixel_values'
  global parse_result
  if nc_coverage == 'ea' or nc_coverage == 'ko':
    parse_result = read_ir105_ea_fast_with_vector(nc_file, highest_step, attr_to_get, conversion_file)
  else:
    parse_result = read_ir105_fd_fast_with_vector(nc_file, highest_step, attr_to_get, conversion_file)
  high_quality_image_name_mono = f'{save_dir}/{Path(out_file).stem}_mono.png'
  high_quality_image_name_color = f'{save_dir}/{Path(out_file).stem}_color.png'
  image_size = IMAGE_SIZE[nc_coverage][highest_step]
  bounds = BOUNDS[nc_coverage]
  processes = []
  # generate_image_from_data_fast(parse_result, high_quality_image_name_mono, image_size, bounds, color_mode='gray')
  p1 = multiprocessing.Process(target=generate_image_from_data_fast, args=(parse_result, high_quality_image_name_mono, image_size, bounds, 'gray'))
  processes.append(p1)
  p1.start()
  p2 = multiprocessing.Process(target=generate_image_from_data_fast, args=(parse_result, high_quality_image_name_color, image_size, bounds, 'color'))
  processes.append(p2)
  p2.start()
  # generate_image_from_data_fast(parse_result, high_quality_image_name_color, image_size, bounds, color_mode='color')
  for p in processes:
    p.join()
  print('saved high quality image[mono]:', high_quality_image_name_mono)
  print('saved high quality image[color]:', high_quality_image_name_color)

  # make highest image using PIL
  print('start downgrade image quality')
  for step in SAVE_IMAGE_STEPS[nc_coverage] :
    try :
      if step == highest_step:
        continue
      out_file = get_json_fname(save_dir, nc_file, step)
      out_image_name_mono = f'{save_dir}/{Path(out_file).stem}_mono.png'
      out_image_name_color = f'{save_dir}/{Path(out_file).stem}_color.png'
      resize_image(high_quality_image_name_mono, out_image_name_mono, IMAGE_SIZE[nc_coverage][step])
      resize_image(high_quality_image_name_color, out_image_name_color, IMAGE_SIZE[nc_coverage][step])
      print('save mono image:', out_image_name_mono)
      print('save mono image:', out_image_name_color)
    except Exception as e :
      print(f"오류 발생 ({nc_file}): {e}")
      continue
    print("waiting for next files...")

if __name__ == '__main__' :
  start_watching(config.WATCH_PATH, None, callback)
  # callback(rf'{watch_path}\2025-03-29\gk2a_ami_le1b_ir105_ea020lc_202503281500_202503290000.nc')
  # callback(rf'{watch_path}\2025-03-29\gk2a_ami_le1b_ir105_fd020ge_202503281500_202503290000.nc')
