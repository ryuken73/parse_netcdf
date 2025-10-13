import os
from pathlib import Path
from parseWithVectorNC import *
from watchFolder_Thread import start_watching
#from to_epsg3857_image import convert_to_equi_rectangle
from to_epsg3857_keep_size import convert_to_equi_rectangle
from config import get_config
from PIL import Image
import sys
import multiprocessing

config = get_config()
print(f'Running in {config.ENV} mode')
print(f'OUT_PATH_RDR = {config.OUT_PATH_RDR}')
print(f'WATCH_PATH_RDR = {config.WATCH_PATH_RDR}')


SAVE_IMAGE_STEPS =  [1, 5, 10]
IMAGE_SIZE = {
  5: (1703, 1956),
  10: (1277, 1467)
}

out_dir = config.OUT_PATH_RDR
highest_step = 1

def callback(rdr_file):
  print(f'processing file {rdr_file}')
  sub_dir = os.path.dirname(rdr_file).split(config.OS_SEP)[-1:][0]
  print(sub_dir)
  save_dir = f"{out_dir}/{sub_dir}"
  os.makedirs(save_dir, exist_ok=True)
  print(f'save file to {save_dir}')
  print(f'start read RDR hsp file')

  colored_array = read_RDR_bin(rdr_file)
  converted_array = reproject_RDR(colored_array)
  print(f'done read RDR hsp file')
  rdr_image_name = f'{save_dir}/{Path(rdr_file).stem}_step{highest_step}.png'
  print('start save image')
  image = Image.fromarray(converted_array.astype(np.uint8), 'RGBA')
  image.save(rdr_image_name)
  print(f'done save image {rdr_image_name}')
  # make equi-rectangle image for unreal texture
  high_quality_image_name_rdr_equi = f'{save_dir}/{Path(rdr_file).stem}_step{highest_step}_equi.png'
  convert_to_equi_rectangle('rdr', rdr_image_name, high_quality_image_name_rdr_equi)
  print(f'done make equi-rectangle image {high_quality_image_name_rdr_equi}')
  high_quality_image_name_rdr_equi_normal = f'{save_dir}/{Path(rdr_file).stem}_step{highest_step}_equi_normal.png'
  create_normal_map_for_rdr(high_quality_image_name_rdr_equi, output_normal_path=high_quality_image_name_rdr_equi_normal)
  print(f'done make normal map image {high_quality_image_name_rdr_equi_normal}')

  print('start downgrade image quality')
  for step in SAVE_IMAGE_STEPS:
    try :
      if step == highest_step:
        continue
      out_image_name = f'{save_dir}/{Path(rdr_file).stem}_step{step}.png'
      resize_image(rdr_image_name, out_image_name, IMAGE_SIZE[step])
      print('save image:', out_image_name)
    except Exception as e :
      print(f"오류 발생 ({rdr_file}): {e}")
      continue
    
  print("waiting for next files...")

if __name__ == '__main__' :
  start_watching(config.WATCH_PATH_RDR, None, callback)
