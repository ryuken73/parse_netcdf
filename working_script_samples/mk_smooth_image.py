from PIL import Image
img = Image.open(r'D:\002.Code\099.study\deck.gl.test\public\output_image_fast_4096_17m_step1.png')
img = img.resize((1400, 1400), Image.Resampling.LANCZOS)
img.save(r'D:\002.Code\099.study\deck.gl.test\public\output_image_fast_4096_17m_1400_step1_smooth.png')