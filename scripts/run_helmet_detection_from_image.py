from ultralytics import YOLO
import cv2
from utils.paths import *
from utils.compute_iou import compute_iou
from utils.get_labels import get_label
from scripts.extract_targets_from_detections import extract_targets_from_detections
from scripts.detect_and_annotate_helmet_compliance import detect_and_annotate_helmet_compliance
      
def run_helmet_detection_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error on load image: {image_path}")
        return
    
    persons, bicycles, motorcycles = extract_targets_from_detections(image_path)

    image_with_detections = detect_and_annotate_helmet_compliance(image_path,persons,bicycles,motorcycles,image)

    cv2.imshow("Helmet detection", image_with_detections)
    cv2.imwrite("sample_6.png",image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



