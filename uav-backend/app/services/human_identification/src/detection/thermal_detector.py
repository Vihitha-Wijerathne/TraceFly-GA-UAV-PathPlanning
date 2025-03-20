import cv2
import numpy as np

class ThermalDetector:
    def __init__(self, threshold_temp=127):  # Lower threshold for better detection
        self.threshold_temp = threshold_temp

    def process_image(self, image_path):
        """Process thermal image and detect human-temperature regions"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold for human body temperature
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and get bounding boxes
        human_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500 and area < 50000:  # Adjust area thresholds
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter by aspect ratio to avoid very thin or wide boxes
                aspect_ratio = float(w)/h
                if 0.3 < aspect_ratio < 3.0:
                    human_detections.append((x, y, x+w, y+h))
        
        # Merge overlapping detections
        human_detections = self._merge_boxes(human_detections)
        
        return img, human_detections

    def _merge_boxes(self, boxes, overlap_thresh=0.3):
        """Merge overlapping bounding boxes"""
        if not boxes:
            return []

        # Convert to numpy array for easier computation
        boxes = np.array(boxes)
        pick = []
        
        # Compute coordinates
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        
        # Compute area
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Find overlap
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlap_thresh)[0])))
        
        return [tuple(map(int, box)) for box in boxes[pick]]

    def draw_detections(self, img, detections):
        """Draw bounding boxes around detected humans"""
        result = img.copy()
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, 'Human', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return result 