import os
from PIL import Image
from pathlib import Path

def resize_image(src_image_path, target_image_path, target_resolution):
  with Image.open(src_image_path) as src_img:
    resized = src_img.resize(target_resolution, Image.Resampling.LANCZOS)
    resized.save(target_image_path)


def convert_png_to_webp(input_path, output_path, quality=80, lossless=False):
    """
    PNG 이미지를 WebP로 변환 (알파 채널 유지).:

    
    Parameters:
    - input_path: 입력 PNG 파일 경로
    - output_path: 출력 WebP 파일 경로
    - quality: WebP 품질 (0-100, 기본값 80)
    - lossless: 무손실 압축 여부 (기본값 False)
    """
    try:
        img = Image.open(input_path)
        
        # 알파 채널 유지
        if img.mode not in ('RGBA', 'RGB'):
            img = img.convert('RGBA' if 'A' in img.mode else 'RGB')
        
        # WebP로 저장
        img.save(output_path, 'WEBP', quality=quality, lossless=lossless)
        print(f"이미지가 {output_path}로 저장되었습니다. 파일 크기: {os.path.getsize(output_path) / 1024:.2f} KB")
    except Exception as e:
        print(f"오류 발생: {e}")

saveDir = r"D:/002.Code/002.node/weather_api/data/weather/rdr/2025-07-07"
directory = Path(f"{saveDir}/orig")

rdr_origFiles = [f.name for f in directory.glob('*.png') if f.is_file()]
for rdr_origFile in rdr_origFiles:
  print(f"resize: {rdr_origFile}")
  resize_image(f'{saveDir}/orig/{rdr_origFile}', f'{saveDir}/{rdr_origFile}', (1277, 1467))
  # resize_image(f'{saveDir}/orig/{rdr_origFile}', f'{saveDir}/{rdr_origFile}', (1277, 1467))
  # convert_png_to_webp(f'{saveDir}/orig/{rdr_origFile}', f'{saveDir}/{Path(rdr_origFile).stem}.webp', quality=80, lossless=False)
  

