from pathlib import Path
from parseNC import *

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

def nc_to_json(step):
  out_dir = './jsonfiles'
  parseResult = []
  conversion_array = np.loadtxt(conversion_mapping_table_file)
  use_index = 1 #mono IR (use second column(brightness temperature))

  directory = Path("./ncfiles_temp")
  nc_files = [f.name for f in directory.glob("*.nc") if f.is_file()]
  print("nc files:", nc_files)
  for nc_file_name in nc_files:
    print(f"처리 중: {nc_file_name}")
    nc_file = f'./ncfiles_temp/{nc_file_name}'
    try :
      attr_to_get = 'image_pixel_values'
      out_file, nc_coverage, nc_projection = mk_out_file_name(nc_file, step, out_dir)
      attr_raw, dim_x, dim_y, projAttrs = get_params_func[nc_projection](nc_file, attr_to_get, GRID_MAPPING.get('no_grid', None))
      parseResult = parse_func[nc_projection](step, dim_x, dim_y, attr_raw, projAttrs, conversion_array, use_index)

      print("parse result Done:", len(parseResult))
      save_to_file(out_file, parseResult)
      out_image_name_mono = f'{out_dir}/{Path(out_file).stem}_mono.png'
      out_image_name_color = f'{out_dir}/{Path(out_file).stem}_color.png'
      save_to_image_ir105(parseResult, out_image_name_mono, nc_coverage, mode='mono')
      save_to_image_ir105(parseResult, out_image_name_color, nc_coverage, mode='color')
      compress_file(out_file)

      # debug result
      lons, lats, attr_values = desctruct_att_lat_lon(parseResult);
      print_result(lons, lats, attr_values, attr_to_get)
      # show_plot(lons, lats, attr_values, nc_coverage, f"visualization: {out_file}")
    except Exception as e :
      print(f"오류 발생 ({nc_file}): {e}")
      continue
    
if __name__ == '__main__' :
  steps = [10, 8, 4, 2, 1]
  for step in steps:
    nc_to_json(step)


