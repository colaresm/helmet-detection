from ultralytics import YOLO
import cv2
from utils.paths import *
from utils.compute_iou import compute_iou
from utils.get_labels import get_label

model1 = YOLO(base_model_path)           
model2 = YOLO(model_path)          

TARGET_CLASS_IDS_MODEL1 = [0, 1, 3]   

def run_helmet_detection_from_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error on load image: {image_path}")
        return

    results1 = model1(image_path, conf=0.45, iou=0.45)
    results2 = model2(image_path, conf=0.45, iou=0.45)

    result1 = results1[0]
    result2 = results2[0]

    boxes1 = result1.boxes.xyxy.cpu().numpy()
    class_ids1 = result1.boxes.cls.cpu().numpy()
    confidences1 = result1.boxes.conf.cpu().numpy()

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
           # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
           # cv2.putText(image, f"{name} {conf:.2f}", (bbox[0], bbox[3] + 20),
                    #    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    class_ids2 = result2.boxes.cls.cpu().numpy()
    confidences2 = result2.boxes.conf.cpu().numpy()
    boxes2 = result2.boxes.xyxy.cpu().numpy()

    for i in range(len(class_ids2)):
        class_id = int(class_ids2[i])
        name = model2.names[class_id]
        conf = confidences2[i]
        bbox_helmet = list(map(int, boxes2[i]))

        label = name  
        found_person = None
        for person in persons:
            iou = compute_iou(bbox_helmet, person)
            print(iou)
            if iou != 0:  
                found_person = person
                break

        
        if  found_person:
            in_motorcycle = any(compute_iou(found_person, b) > 0.2 for b in motorcycles)
            in_bike = any(compute_iou(found_person, b) > 0 for b in bicycles)
            has_helmet = name =="with helmet"
            label = get_label(in_motorcycle,in_bike,has_helmet)

            x1, y1, x2, y2 = map(int, boxes2[i])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("Helmet detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



