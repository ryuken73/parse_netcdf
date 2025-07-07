import os
from pathlib import Path
from parseWithVectorNC import *
from watchFolder_Thread import start_watching
from config import get_config
from PIL import Image
import sys
import multiprocessing

config = get_config()
print(f'Running in {config.ENV} mode')
print(f'OUT_PATH_RDR = {config.OUT_PATH_RDR}')
print(f'WATCH_PATH_RDR = {config.WATCH_PATH_RDR}')

out_dir = config.OUT_PATH_RDR

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
  rdr_image_name = f'{save_dir}/{Path(rdr_file).stem}.png'
  print('start save image')
  image = Image.fromarray(converted_array.astype(np.uint8), 'RGBA')
  image.save(rdr_image_name)
  print(f'done save image {rdr_image_name}')
  print("waiting for next files...")

if __name__ == '__main__' :
  start_watching(config.WATCH_PATH_RDR, None, callback)
