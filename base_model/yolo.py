from ultralytics import YOLO
import cv2

model = YOLO('yolov8s.pt')
result = model('cat.png', show = True)

cv2.waitKey(0)