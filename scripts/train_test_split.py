import os
import random
import shutil
from utils.paths import*

def create_train_val_split():
    for dir in [train_img_path, val_img_path, train_labels_path, val_labels_path]:
        os.makedirs(dir, exist_ok=True)

    images = [img[:-4] for img in os.listdir(images_path)]
    random.shuffle(images)
    split = int(0.8 * len(images))

    count_total = len(images)
    count_train = 0
    count_val = 0
    count_ignored = 0

    for i in range(len(images)):
        if not os.path.exists(os.path.join(labels_path, f"{images[i]}.txt")):
            count_ignored += 1
            count_total -= 1
            continue

        if i < split:
            shutil.copy(os.path.join(images_path, f"{images[i]}.png"), train_img_path)
            shutil.copy(os.path.join(labels_path, f"{images[i]}.txt"), train_labels_path)
            count_train += 1
        else:
            shutil.copy(os.path.join(images_path, f"{images[i]}.png"), val_img_path)
            shutil.copy(os.path.join(labels_path, f"{images[i]}.txt"), val_labels_path)
            count_val += 1

        count_total -= 1
        print(f"\rImages: {count_total} >> Train: {count_train} | Val: {count_val} | Ignored: {count_ignored}     ", end='', flush=True)
