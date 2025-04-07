from ultralytics import YOLO
import cv2
from utils.paths import *
from scripts.extract_targets_from_detections import extract_targets_from_detections
from scripts.detect_and_annotate_helmet_compliance import detect_and_annotate_helmet_compliance

def run_helmet_detection_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error on load video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break   

        persons, bicycles, motorcycles = extract_targets_from_detections(frame)

        frame_with_detections = detect_and_annotate_helmet_compliance(frame,persons,bicycles,motorcycles,frame)

        cv2.imshow("Helmet detection", frame_with_detections)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
