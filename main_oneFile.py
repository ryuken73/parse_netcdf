from pathlib import Path
from parseNC import *

GRID_MAPPING = {
  "kma_grid": 'gk2a_imager_projection', # if data is ctps, grid_mapping = gk2a_imager_projection
  "no_grid": None
}

def nc_to_json(step):
  out_dir = './jsonfiles'
  parseResult = []
  conversion_array = np.loadtxt('ir105_conversion_c.txt');
  use_index = 1 #mono IR

  directory = Path("./ncfiles")
  nc_files = [f.name for f in directory.glob("*.nc") if f.is_file()]
  print("nc files:", nc_files)
  for nc_file_name in nc_files:
    print(f"처리 중: {nc_file_name}")
    nc_file = f'./ncfiles/{nc_file_name}'
    try :
      attr_to_get = 'image_pixel_values'
      out_file, nc_coverage, nc_projection = mk_out_file_name(nc_file, step, out_dir)
      attr_raw, dim_x, dim_y, projAttrs = get_params_func[nc_projection](nc_file, attr_to_get, GRID_MAPPING.get('no_grid', None))
      parseResult = parse_func[nc_projection](step, dim_x, dim_y, attr_raw, projAttrs, conversion_array, use_index)

      print("parse result Done:", len(parseResult))
      save_to_file(out_file, parseResult)
      compress_file(out_file)

      # debug result
      lons, lats, attr_values = desctruct_att_lat_lon(parseResult);
      print_result(lons, lats, attr_values, attr_to_get)
      # show_plot(lons, lats, attr_values, nc_coverage, f"visualization: {out_file}")
    except Exception as e :
      print(f"오류 발생 ({nc_file}): {e}")
      continue
    
if __name__ == '__main__' :
  steps = [10, 8, 5, 4]
  for step in steps:
    nc_to_json(step)


