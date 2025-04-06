import argparse
import sys
from scripts.global_imports import*

parser = argparse.ArgumentParser(description='Detect helmets in an image or video.')

parser.add_argument('file_type', type=str, choices=['image', 'video'], help='Input type: "image" or "video"')
parser.add_argument('file_path', type=str, help='File path, ex: data/test_data/test_image.png')

args = parser.parse_args()

is_image = args.file_type == 'image'
file_path = args.file_path

if is_image:  
  run_helmet_detection_from_image(file_path)
else:
  run_helmet_detection_from_video(file_path)

#python test.py image /caminho/para/arquivo.jpg