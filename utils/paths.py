import os

project_path = os.getcwd()
dataset_path = os.path.join(project_path, 'helmet_detection_yolo')

images_path = dataset_path+"/data/images"
annotations_path =  dataset_path+"/data/annotations"
labels_path = os.path.join(dataset_path, "labels")

train_img_path = os.path.join(dataset_path, "train", "images")
train_labels_path = os.path.join(dataset_path, "train", "labels")
val_img_path = os.path.join(dataset_path, "val", "images")
val_labels_path = os.path.join(dataset_path, "val", "labels")
test_img_path = os.path.join(dataset_path, "test", "images")

model_path = "models/yolov8n_helmet_detection.pt"
base_model_path = "models/yolov8n.pt"
config_file_path = f"{dataset_path}/config.yaml"

