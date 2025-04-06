import cv2
from ultralytics import YOLO
from utils.paths import *

def run_helmet_detection_from_video(video_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error on load video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Helmet Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
