from to_epsg3857_image import covert_to_equi_rectangle

coverage = 'ea'
in_file = '/data/node_project/weather_data/out_data/gk2a/2025-09-23/gk2a_ami_le1b_ir105_ea020lc_202509231450_202509232350_step1_color.png'
out_file = 'gk2a_ami_le1b_ir105_ea020lc_202509231450_202509232350_step1_color_equi.png'

covert_to_equi_rectangle(coverage, in_file, out_file)
