from PIL import Image
img = Image.open(r'D:\002.Code\001.python\netcdf\working_script_samples\output_image_fast_w_vector_3192_5M.png')
img2048 = img.resize((2048, 2048), Image.Resampling.LANCZOS)
img1400 = img.resize((1400, 1400), Image.Resampling.LANCZOS)
img2048.save(r'D:\002.Code\001.python\netcdf\working_script_samples\output_image_fast_w_vector_3192_5M_to_2048.png')
img1400.save(r'D:\002.Code\001.python\netcdf\working_script_samples\output_image_fast_w_vector_3192_5M_to_1400.png')