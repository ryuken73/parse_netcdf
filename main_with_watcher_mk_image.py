import os
from pathlib import Path
from parseWithVectorNC import *
from watchFolder_Thread import start_watching
import sys

SAVE_IMAGE_STEPS = {
  'ea': [1, 5, 10],
  'fd': [1, 5, 10]
}
IMAGE_SIZE = {
  'ea': {
    1: (1500, 1300),
    5: (1200, 1040),
    10: (900, 780)
  },
  'fd': {
    1: (3192, 3192),
    5: (2048, 2048),
    10: (1400, 1400),
  }
}
BOUNDS ={
  'ea': [76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447],
  'fd': [60, -80, 180, 80]
}

out_dir = 'D:/002.Code/002.node/weather_api/data/weather/gk2a'
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

def callback(nc_file):
  print(f'processing file {nc_file}')
  sub_dir = os.path.dirname(nc_file).split('\\')[-1:][0]
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
  if nc_coverage == 'ea':
    parse_result = read_ir105_ea_fast_with_vector(nc_file, highest_step, attr_to_get, conversion_file)
  else:
    parse_result = read_ir105_fd_fast_with_vector(nc_file, highest_step, attr_to_get, conversion_file)
  high_quality_image_name_mono = f'{save_dir}/{Path(out_file).stem}_mono.png'
  high_quality_image_name_color = f'{save_dir}/{Path(out_file).stem}_color.png'
  image_size = IMAGE_SIZE[nc_coverage][highest_step]
  bounds = BOUNDS[nc_coverage]
  generate_image_from_data_fast(parse_result, high_quality_image_name_mono, image_size, bounds, color_mode='gray')
  print('save high quality image[mono]:', high_quality_image_name_mono)
  generate_image_from_data_fast(parse_result, high_quality_image_name_color, image_size, bounds, color_mode='color')
  print('save high quality image[color]:', high_quality_image_name_color)

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

# def nc_to_json(step):
#   out_dir = './jsonfiles'
#   parseResult = []
#   use_index = 1 #mono IR (use second column(brightness temperature))

#   directory = Path("./ncfiles_temp")
#   nc_files = [f.name for f in directory.glob("*.nc") if f.is_file()]
#   print("nc files:", nc_files)
#   for nc_file_name in nc_files:
#     print(f"처리 중: {nc_file_name}")
#     nc_file = f'./ncfiles_temp/{nc_file_name}'
#     try :
#       attr_to_get = 'image_pixel_values'
#       out_file, nc_coverage, nc_projection = mk_out_file_name(nc_file, step, out_dir)
#       attr_raw, dim_x, dim_y, projAttrs = get_params_func[nc_projection](nc_file, attr_to_get, GRID_MAPPING.get('no_grid', None))
#       parseResult = parse_func[nc_projection](step, dim_x, dim_y, attr_raw, projAttrs, conversion_array, use_index)

#       print("parse result Done:", len(parseResult))
#       save_to_file(out_file, parseResult)
#       if step in SAVE_IMAGE_STEPS[target_coverage]:
#         out_image_name_mono = f'{out_dir}/{Path(out_file).stem}_mono.png'
#         out_image_name_color = f'{out_dir}/{Path(out_file).stem}_color.png'
#         save_to_image_ir105(parseResult, out_image_name_mono, nc_coverage, step, mode='mono')
#         save_to_image_ir105(parseResult, out_image_name_color, nc_coverage, step, mode='color')
#       compress_file(out_file)

#       # debug result
#       lons, lats, attr_values = desctruct_att_lat_lon(parseResult);
#       print_result(lons, lats, attr_values, attr_to_get)
#       # show_plot(lons, lats, attr_values, nc_coverage, f"visualization: {out_file}")
#     except Exception as e :
#       print(f"오류 발생 ({nc_file}): {e}")
#       continue
    
if __name__ == '__main__' :
  watch_path = r"D:\002.Code\002.node\weather_api\kma_fetch\data\weather\gk2a"
  start_watching(watch_path, None, callback)
  # callback(rf'{watch_path}\2025-03-29\gk2a_ami_le1b_ir105_ea020lc_202503281500_202503290000.nc')
  # callback(rf'{watch_path}\2025-03-29\gk2a_ami_le1b_ir105_fd020ge_202503281500_202503290000.nc')
