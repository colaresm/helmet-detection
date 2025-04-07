from ultralytics import YOLO
from utils.paths import*

TARGET_CLASS_IDS_MODEL1 = [0, 1, 3]   

def extract_targets_from_detections(source):
    base_model = YOLO(base_model_path)           
    persons = []
    bicycles = []
    motorcycles = []
    class_ids_base_model,boxes_base_model = detect_objects_with_base_model(base_model,source)

    for i in range(len(class_ids_base_model)):
        class_id = int(class_ids_base_model[i])
        bbox = list(map(int, boxes_base_model[i]))
        if class_id in TARGET_CLASS_IDS_MODEL1:
            if class_id == 0:
                persons.append(bbox)
            elif class_id == 1:
                bicycles.append(bbox)
            elif class_id == 3:
                motorcycles.append(bbox)
        
    return persons, bicycles, motorcycles

def detect_objects_with_base_model(base_model,source):

    detections_base_model = base_model(source = source, conf=0.45, iou=0.45)
    detections_base_model = detections_base_model[0]
    
    boxes_base_model = detections_base_model.boxes.xyxy.cpu().numpy()
    class_ids_base_model = detections_base_model.boxes.cls.cpu().numpy()
#   confidences_base_model = detections_base_model.boxes.conf.cpu().numpy()

    return class_ids_base_model,boxes_base_model