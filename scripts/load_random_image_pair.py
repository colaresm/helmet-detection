import os
import random

def load_random_image_pair(images_path,annotations_path):

    img_name = random.choice(os.listdir(images_path))
    img_file = os.path.join(images_path, img_name)
    xml_file = os.path.join(annotations_path, img_name[:-4]+'.xml')

    return img_file, xml_file