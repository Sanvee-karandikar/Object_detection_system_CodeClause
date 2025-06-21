from ultralytics import YOLO

#Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt") #'n' means nano(small and fast model