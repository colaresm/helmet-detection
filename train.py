import os
from ultralytics import YOLO
from utils.paths import*
from config.create_config_file import*
from scripts.global_imports import*


create_labels(annotations_path, labels_path)

create_train_val_split()

create_config_file()


yolo = YOLO('models/yolov8n.pt')   
yolo.train(
     data=config_file_path,
     epochs=320, 
     patience=20,  
     batch=-1,  
     save_period=10, 
     dropout=0.2,  
     plots=True  
)

yolo.save(model_path)
