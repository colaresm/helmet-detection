from ultralytics import YOLO
import cv2
from utils.paths import *


def run_helmet_detection_from_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error on load image: {image_path}")
        return

    model = YOLO(model_path)
    model(image_path, show=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
