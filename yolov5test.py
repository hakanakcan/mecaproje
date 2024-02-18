import os
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import Annotator, colors
import cv2

BASE_DIR = 'C:/Users/Hakan/Desktop/Projetest'

VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')
RUNS_DIR = os.path.join(BASE_DIR, 'runs')

video_path = os.path.join(VIDEOS_DIR, '5.mp4')
video_path_out = os.path.join(VIDEOS_DIR, '5_out_4.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join(RUNS_DIR, 'detect', 'train9', 'weights', 'last.pt')

# Load a model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = attempt_load(model_path, map_location=device)
model.to(device).eval()

threshold = 0.5

while ret:
    img = torch.from_numpy(frame).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, threshold, 0.4)

    # Process detections
    for det in pred[0]:
        x1, y1, x2, y2, conf, class_id = det
        label = model.names[int(class_id)]
        box = [x1, y1, x2, y2, conf, class_id]
        Annotator.add(frame, label, box, color=colors(int(class_id), True), xyxy=True)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
