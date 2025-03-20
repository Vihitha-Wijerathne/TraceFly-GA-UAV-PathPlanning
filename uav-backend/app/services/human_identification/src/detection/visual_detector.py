from ultralytics import YOLO
import cv2
import numpy as np

class VisualDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize YOLOv8 model"""
        self.model = YOLO(model_path)
        
    def detect_people(self, image_path):
        """Detect people in images using YOLOv8"""
        results = self.model(image_path)
        # Filter for person class (class 0 in COCO)
        people_detections = results[0].boxes[results[0].boxes.cls == 0]
        # Convert to list of (x1, y1, x2, y2) coordinates
        detections = []
        for det in people_detections:
            box = det.xyxy[0].cpu().numpy()
            detections.append(tuple(map(int, box)))
        return detections
    
    def draw_detections(self, image_path, detections):
        """Draw bounding boxes around detected people"""
        img = cv2.imread(image_path)
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(img, (x1, y1), (x2, y2), 
                        (0, 255, 0), 2)
            cv2.putText(img, 'Person', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img 