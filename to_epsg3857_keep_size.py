from pyproj import Transformer, CRS
import subprocess
import tempfile

# WGS84 (위경도, EPSG:4326) 객체 생성
# proj_wgs84 = Proj(init='epsg:4326') # 최신 pyproj 버전에서는 Proj(proj='epsg:4326') 사용 권장
# EPSG:3857 (구글 메르카토르) 객체 생성
# proj_epsg3857 = Proj(init='epsg:3857') # 최신 pyproj 버전에서는 Proj(proj='epsg:3857') 사용 권장

# 참고: 최신 pyproj 버전 (v2 이상)에서는 `init` 대신 `proj` 인자를 사용하며, `Proj` 객체 생성 시좌표계 코드를 직접 지정합니다.
wgs84_values = {
  "fd":{
    "UL": (0, 80),
    "LR": (190, -80),
    "width": 3192,
    "height": 3192,
  },
  "ea":{
    "UL": (76.8111834, 61.9310477),
    "LR": (175.112668, 11.3541175),
    "width": 1500,
    "height": 1300,
  },
  "ko":{
    "UL": (113.99641, 45.729865),
    "LR": (138.003582, 29.312252),
    "width": 800,
    "height": 800,
  },
  "rdr":{
    "UL": (118.8394260710767, 43.572496647155695),
    "LR": (133.5627133041138, 30.102047565010807),
    "width": 2554,
    "height": 2934,
  },
  "aws":{
    "UL": (124.4, 38.6),
    "LR": (131.6, 33.0),
    "width": 2500,
    "height": 2400,
  },
}

def get_xy_from_latlng(lat, lng):
  web_mercator_crs = CRS.from_epsg("3857")
  wgs84_crs = CRS.from_epsg("4326")
  transformer_to_mercator = Transformer.from_crs(wgs84_crs, web_mercator_crs)
  x_3857, y_3857 = transformer_to_mercator.transform(lng, lat);
  return x_3857, y_3857  

def get_wgs84_boundary_coords(coverage):
  (lat_U, lng_U) = wgs84_values[coverage]["UL"]
  (lat_L, lng_L) = wgs84_values[coverage]["LR"]
  return lat_U, lng_U, lat_L, lng_L

def run_gdal_translate(in_file, out_file, ulx, uly, lrx, lry):
  cmd = f"gdal_translate -of Gtiff -a_srs EPSG:3857 -a_ullr {ulx} {uly} {lrx} {lry} {in_file} {out_file}"
  out = subprocess.run(cmd , shell=True, capture_output=True, text=True, check=True)
  print(out.stdout)

def run_gdalwarp(in_file, out_file, xmin=-180, xmax=180, width=7200, height=3600):
  cmd = f"gdalwarp -t_srs EPSG:4326 -te {xmin} -90 {xmax} 90 -ts {width} {height} -dstalpha {in_file} {out_file}"
  out = subprocess.run(cmd , shell=True, capture_output=True, text=True, check=True)
  print(out.stdout)

def run_gdalwarp_keep_size(in_file, out_file, options):
  xmin = options["xmin"]
  xmax = options["xmax"]
  ymin = options["ymin"]
  ymax = options["ymax"]
  width = options["width"]
  height = options["height"]
  cmd = f"gdalwarp -t_srs EPSG:4326 -te {xmin} {ymin} {xmax} {ymax} -ts {width} {height} -dstalpha {in_file} {out_file}"
  out = subprocess.run(cmd , shell=True, capture_output=True, text=True, check=True)
  print(out.stdout)


def covert_to_equi_rectangle(coverage, in_file, out_file):
  lat_U, lng_U, lat_L, lng_L = get_wgs84_boundary_coords(coverage);
  ulx,uly = get_xy_from_latlng(lat_U, lng_U)
  lrx,lry = get_xy_from_latlng(lat_L, lng_L)
  width = wgs84_values[coverage]["width"]
  height = wgs84_values[coverage]["height"]
  print(f"위경도 ({lat_U}, {lng_U}, {lat_L}, {lng_L}) ")
  print(f"EPSG:3857 좌표: ({ulx}, {uly}, {lrx}, {lry})")
  options = {
    "xmin": lat_U,
    "xmax": lat_L,
    "ymin": lng_L,
    "ymax": lng_U,
    "width": width,
    "height": height
  }
  try:
    with tempfile.NamedTemporaryFile() as temp_file:
      print(f"use tempfile {temp_file.name}")
      if coverage == 'fd':
        print("in fd, lat over 180 make minus lrx and eventually converted image is invalid. use plus value for lrx")
        lrx = 21150703.250721980
      run_gdal_translate(in_file, temp_file.name, ulx, uly, lrx, lry)
      run_gdalwarp_keep_size(temp_file.name, out_file, options)
    print(f"convert done - in: {in_file}")
    print(f"convert done - to: {out_file}")
  except Exception as e:
    print(e)
    print(f"error to convert file")

#in_file = '/data/node_project/weather_data/out_data/gk2a/2025-09-23/gk2a_ami_le1b_ir105_fd020ge_202509231450_202509232350_step1_color.png'
#in_file = '/data/node_project/weather_data/out_data/gk2a/2025-09-23/gk2a_ami_le1b_ir105_ko020lc_202509231458_202509232358_step1_color.png'
#in_file = '/data/node_project/weather_data/out_data/aws/2025-09-13/AWS_MIN_202509132330_RN_24HR_step1.png'
#in_file = '/data/node_project/weather_data/out_data/rdr/2025-09-26/RDR_CMP_HSP_PUB_202509260845_step1.png'
#out_file = 'gk2a_ami_le1b_ir105_fd020ge_202509231450_202509232350_step1_color_keep_size_equi.png'
#out_file = 'gk2a_ami_le1b_ir105_ko020lc_202509231458_202509232358_step1_color_euqi_keep_size.png'
#out_file = 'AWS_MIN_202509132330_RN_24HR_step1_keep_size_equi.png'
#out_file = 'RDR_CMP_HSP_PUB_202509260845_step1_keep_size.png'

#covert_to_equi_rectangle('rdr', in_file, out_file)
