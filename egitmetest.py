from ultralytics import YOLO
from smpy import python
from yolov5 import train


# Load a model
model = YOLO("yolov5s.yaml")  # build a new model from scratch

# Use the model
#results = model.train(data="project.yaml", epochs=300)  # train the model

# Train YOLOv5s on COCO128 for 3 epochs
python train.py --img 640 --batch 16 --epochs 3 --data project.yaml --weights yolov5s.pt
