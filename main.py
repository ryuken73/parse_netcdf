from pathlib import Path
from parseNC import *

if __name__ == '__main__' :
  step = 8 
  out_dir = './jsonfiles'
  parseResult = []

  directory = Path("./ncfiles")
  nc_files = [f.name for f in directory.glob("*.nc") if f.is_file()]
  print("nc files:", nc_files)
  for nc_file_name in nc_files:
    print(f"처리 중: {nc_file_name}")
    nc_file = f'./ncfiles/{nc_file_name}'
    try :
      attr_to_get = 'image_pixel_values'
      out_file, nc_coverage = mk_out_file_name(nc_file, step, out_dir)
      attr_raw, dim_x, dim_y, projAttrs = get_params_lc(nc_file, attr_to_get, GRID_MAPPING.get('ir105', None))
      parseResult = parseLc(step, dim_x, dim_y, attr_raw, projAttrs)

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



