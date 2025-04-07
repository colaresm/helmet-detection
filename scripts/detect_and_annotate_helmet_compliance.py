from ultralytics import YOLO
import cv2
from utils.paths import*
from utils.compute_iou import*
from utils.get_labels import*

def detect_and_annotate_helmet_compliance(source,persons,bicycles,motorcycles,image):
    helmet_detect_model = YOLO(model_path)       
    
    detections_helmet_model = helmet_detect_model(source=source, conf=0.45, iou=0.45)             
    detections_helmet_model = detections_helmet_model[0]
    classes_helmet_model = detections_helmet_model.boxes.cls.cpu().numpy()
    confidences_helmet_model = detections_helmet_model.boxes.conf.cpu().numpy()
    boxes_helmet_model = detections_helmet_model.boxes.xyxy.cpu().numpy()

    for i in range(len(classes_helmet_model)):
        class_id = int(classes_helmet_model[i])
        name = helmet_detect_model.names[class_id]
        conf = confidences_helmet_model[i]
        bbox_helmet = list(map(int, boxes_helmet_model[i]))

        label = name  
        found_person = None

        for person in persons:
            iou = compute_iou(bbox_helmet, person)
            if iou != 0:  
                found_person = person
                break

        if  found_person:
            in_motorcycle = any(compute_iou(found_person, b) > 0.2 for b in motorcycles)
            in_bike = any(compute_iou(found_person, b) > 0 for b in bicycles)
            has_helmet = name == "with helmet"
            label = get_label(in_motorcycle,in_bike,has_helmet)

            x1, y1, x2, y2 = map(int, boxes_helmet_model[i])
            cv2.rectangle(image, (x1, y1), (x2, y2), get_color(has_helmet), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_color(has_helmet), 2)
    
    return image

def get_color(has_helmet):
    if has_helmet:
        return (0,255,0)
    return (0,0,255)