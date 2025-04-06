from ultralytics import YOLO
import cv2
from utils.paths import *
from utils.compute_iou import compute_iou
from utils.get_labels import get_label

model1 = YOLO(base_model_path)  
model2 = YOLO(model_path)        

TARGET_CLASS_IDS_MODEL1 = [0, 1, 3]  

def run_helmet_detection_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro on load video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break   

        results1 = model1.predict(source=frame, conf=0.45, iou=0.45, verbose=False)[0]
        results2 = model2.predict(source=frame, conf=0.45, iou=0.45, verbose=False)[0]

        boxes1 = results1.boxes.xyxy.cpu().numpy()
        class_ids1 = results1.boxes.cls.cpu().numpy()
        confidences1 = results1.boxes.conf.cpu().numpy()

        persons = []
        bicycles = []
        motorcycles = []

        for i in range(len(class_ids1)):
            class_id = int(class_ids1[i])
            bbox = list(map(int, boxes1[i]))
            if class_id in TARGET_CLASS_IDS_MODEL1:
                if class_id == 0:
                    persons.append(bbox)
                elif class_id == 1:
                    bicycles.append(bbox)
                elif class_id == 3:
                    motorcycles.append(bbox)

                name = model1.names[class_id]
                conf = confidences1[i]
              #  cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (200, 200, 200), 2)
               # cv2.putText(frame, f"{name} {conf:.2f}", (bbox[0], bbox[1] - 10),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        class_ids2 = results2.boxes.cls.cpu().numpy()
        confidences2 = results2.boxes.conf.cpu().numpy()
        boxes2 = results2.boxes.xyxy.cpu().numpy()

        for i in range(len(class_ids2)):
            class_id = int(class_ids2[i])
            name = model2.names[class_id]
            conf = confidences2[i]
            bbox_helmet = list(map(int, boxes2[i]))

            label = name  
            found_person = None
            for person in persons:
                iou = compute_iou(bbox_helmet, person)
                if iou != 0:  
                    found_person = person
                    break

            if found_person:
                in_motorcycle = any(compute_iou(found_person, b) > 0.2 for b in motorcycles)
                in_bike = any(compute_iou(found_person, b) > 0 for b in bicycles)
                has_helmet = name == "with helmet"
                label = get_label(in_motorcycle, in_bike, has_helmet)

                x1, y1, x2, y2 = bbox_helmet
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imshow("Helmet detection (video)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
