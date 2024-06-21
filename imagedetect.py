#yolov8 image detection test
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load a COCO-pretrained YOLOv8n model
model = YOLO('best(epoch100).pt')

results = model.predict("60.jpg")

res_plotted = results[0].plot()

plt.figure(figsize = (12,12))
plt.imshow(cv2.cvtColor(res_plotted,cv2.COLOR_BGR2RGB))
plt.show()