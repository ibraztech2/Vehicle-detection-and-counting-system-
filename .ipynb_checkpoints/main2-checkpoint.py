import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
frame = "download.jpg" 
frame = cv.imread(frame, 1)
detections = model(frame, 0.5)
for detection in  detections:
    boxes = detection.boxes
    class_name = boxes.cls.cpu().numpy()
    bbox = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    detection = np.concatenate((bbox, conf[:, None]),axis=1)
    print(detection.shape)

    
    
    #if len(class_name) != 0:
        
      #  for  idx, item in enumerate(class_name):
    
