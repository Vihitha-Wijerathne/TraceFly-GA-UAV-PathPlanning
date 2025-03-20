from src.detection.thermal_detector import ThermalDetector
from src.detection.visual_detector import VisualDetector
from src.detection.audio_detector import AudioDetector

class DetectionFusion:
    def __init__(self):
        self.thermal_detector = ThermalDetector()
        self.visual_detector = VisualDetector()
        self.audio_detector = AudioDetector()
        
    def process_scene(self, thermal_path, visual_path, audio_path):
        """Process a scene using all detection modalities"""
        try:
            # Process thermal
            thermal_img, thermal_detections = self.thermal_detector.process_image(thermal_path)
            
            # Process visual
            visual_detections = self.visual_detector.detect_people(visual_path)
            
            # Process audio
            audio_alert = self.audio_detector.predict(audio_path)
            
            return {
                'thermal_detections': len(thermal_detections),
                'visual_detections': len(visual_detections),
                'audio_alert': audio_alert,
                'thermal_img': thermal_img,
                'thermal_contours': thermal_detections,
                'visual_detections': visual_detections
            }
            
        except Exception as e:
            raise RuntimeError(f"Error processing scene: {str(e)}") 