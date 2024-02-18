from ultralytics import YOLO

# Load a model
model = YOLO("yolov5s.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="project.yaml", epochs=300, batch=32)  # train the model

