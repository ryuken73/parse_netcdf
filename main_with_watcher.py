from pathlib import Path
from parseNC import *
from watchFolder_Thread import start_watching
import sys

target_coverage = 'ea' if len(sys.argv) == 1 else sys.argv[1]
print(f'watch filename with _{target_coverage}.')

STEPS = {
  'ea': [4, 8, 10],
  'fd': [4, 8, 10],
}
SAVE_IMAGE_STEPS = {
  'ea': [4],
  'fd': [4]
}

GRID_MAPPING = {
  "kma_grid": 'gk2a_imager_projection', # if data is ctps, grid_mapping = gk2a_imager_projection
  "no_grid": None
}

get_params_func = {
  'lc': get_params_lc,
  'ge': get_params_geos
}

parse_func = {
  'lc': parseLc,
  'ge': parseGeos
}

conversion_mapping_table_file = 'ir105_conversion_c.txt'

conversion_array = np.loadtxt(conversion_mapping_table_file)

# steps = [3, 10, 8, 4]
# steps = [1]
out_dir = 'D:/002.Code/002.node/weather_api/data/weather/gk2a'
use_index = 1

## nc 1개에 대해서 아래 파일들 생성
## ea 파일 (json + image)
##  gzip: step4, step8, step10 
##  mono png: step1(1500X1300, 1분, 1M), convert_1200X1040, convert_900X780
##  color png: step1(1500X1300, 4분, 2M),convert_1200X1040, convert_900X780
## fd 파일 (image only)
##  gzip: none
##  mono png: step1(3192X3192, 2분, 6M),  convert_2048, convert_1400
##  color png: step1(3192X3192, 5분, 10M),  convert_2048, convert_1400

def callback(nc_file):
  steps = STEPS[target_coverage]
  for step in steps:
    print(f'processing file {nc_file}')
    sub_dir = os.path.dirname(nc_file).split('\\')[-1:][0]
    print(f'sub_dir {sub_dir}')
    save_dir = f"{out_dir}/{sub_dir}"
    os.makedirs(save_dir, exist_ok=True)
    print(f'save file to {save_dir}')

    try :
      attr_to_get = 'image_pixel_values'
      out_file, nc_coverage, nc_projection = mk_out_file_name(nc_file, step, save_dir)
      attr_raw, dim_x, dim_y, projAttrs = get_params_func[nc_projection](nc_file, attr_to_get, GRID_MAPPING.get('no_grid', None))
      parseResult = parse_func[nc_projection](step, dim_x, dim_y, attr_raw, projAttrs, conversion_array, use_index)

      print("parse result Done:", len(parseResult))
      save_to_file(out_file, parseResult)
      # if step in SAVE_IMAGE_STEPS[nc_coverage]:
      #   out_image_name_mono = f'{save_dir}/{Path(out_file).stem}_mono.png'
      #   out_image_name_color = f'{save_dir}/{Path(out_file).stem}_color.png'
      #   save_to_image_ir105(parseResult, out_image_name_mono, nc_coverage, step, mode='mono')
      #   save_to_image_ir105(parseResult, out_image_name_color, nc_coverage, step, mode='color')
      compress_file(out_file)

      # debug result
      lons, lats, attr_values = desctruct_att_lat_lon(parseResult);
      print_result(lons, lats, attr_values, attr_to_get)
      # show_plot(lons, lats, attr_values, nc_coverage, f"visualization: {out_file}")
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
  start_watching(watch_path, target_coverage, callback)



