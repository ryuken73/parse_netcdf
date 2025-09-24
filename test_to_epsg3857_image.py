from to_epsg3857_image import covert_to_equi_rectangle

coverage = 'ko'
in_file = '/data/node_project/weather_data/out_data/gk2a/2025-09-25/gk2a_ami_le1b_ir105_ko020lc_202509242302_202509250802_step1_color.png'
out_file = '/data/node_project/weather_data/out_data/gk2a/2025-09-25/gk2a_ami_le1b_ir105_ko020lc_202509242302_202509250802_step1_color_equi.png'

covert_to_equi_rectangle(coverage, in_file, out_file)
