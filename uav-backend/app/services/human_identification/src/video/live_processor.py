import cv2
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
from src.detection.visual_detector import VisualDetector
from src.detection.audio_detector import AudioDetector
import urllib.request
import ssl
import certifi
import sounddevice as sd
import threading
import queue

class LiveProcessor:
    def __init__(self):
        self.visual_detector = VisualDetector()
        self.audio_detector = AudioDetector()
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Audio processing parameters
        self.RATE = 16000
        self.BLOCKSIZE = 1024
        self.CHANNELS = 1
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.last_audio_level = 0
        self.audio_threshold = 0.3  # Increased threshold
        self.noise_floor = 0.05     # Minimum level to consider as actual sound
        # Rolling average for noise estimation
        self.noise_history = []
        self.noise_window = 50      # Number of frames for noise estimation
        
        # Emotion labels
        self.emotions = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'fear']
        
        # Add emotion history for smoothing
        self.emotion_history = {}
        self.history_length = 10  # Shorter history for quicker response
        self.smoothing_alpha = 0.3  # More weight to current emotion
        
        self.emotion_colors = {
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'neutral': (0, 255, 0),    # Green
            'surprise': (255, 0, 255),  # Purple
            'fear': (0, 165, 255)      # Orange
        }

    def smooth_emotions(self, emotions, person_id):
        """Apply temporal smoothing to emotion values"""
        if emotions is None:
            return None

        # Initialize history for new person
        if person_id not in self.emotion_history:
            self.emotion_history[person_id] = []

        # Add current emotions to history
        self.emotion_history[person_id].append(emotions)
        
        # Keep only recent history
        if len(self.emotion_history[person_id]) > self.history_length:
            self.emotion_history[person_id].pop(0)
        
        # Calculate smoothed emotions
        smoothed_emotions = {}
        for emotion in emotions.keys():
            values = [frame[emotion] for frame in self.emotion_history[person_id]]
            smoothed_emotions[emotion] = sum(values) / len(values)
        
        return smoothed_emotions

    def get_facial_features(self, shape):
        """Extract facial features from landmarks"""
        # Convert shape to numpy array
        shape = face_utils.shape_to_np(shape)
        
        # Calculate features
        features = {}
        
        # Mouth features (landmarks 48-68)
        mouth_pts = shape[48:68]
        mouth_width = np.linalg.norm(mouth_pts[6] - mouth_pts[0])
        mouth_height = np.linalg.norm(mouth_pts[3] - mouth_pts[9])
        features['mouth_ratio'] = mouth_height / mouth_width
        
        # Eye features (landmarks 36-48)
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5])
        right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5])
        features['eye_height'] = (left_eye_height + right_eye_height) / 2
        
        # Eyebrow features (landmarks 17-27)
        left_brow = shape[17:22]
        right_brow = shape[22:27]
        brow_height = (np.mean(left_brow[:, 1]) + np.mean(right_brow[:, 1])) / 2
        features['brow_height'] = brow_height
        
        return features

    def calculate_face_metrics(self, landmarks, image_shape):
        """Calculate facial metrics from landmarks"""
        metrics = {}
        
        # Convert landmarks to numpy array
        points = np.array([[lm.x * image_shape[1], lm.y * image_shape[0]] 
                          for lm in landmarks.landmark])
        
        # Mouth metrics
        mouth_top = points[13]
        mouth_bottom = points[14]
        mouth_left = points[78]
        mouth_right = points[308]
        
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        metrics['mouth_ratio'] = mouth_height / mouth_width
        
        # Eye metrics
        left_eye_top = points[159]
        left_eye_bottom = points[145]
        right_eye_top = points[386]
        right_eye_bottom = points[374]
        
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        metrics['eye_openness'] = (left_eye_height + right_eye_height) / 2
        
        # Eyebrow metrics
        left_brow = points[66]
        right_brow = points[296]
        nose_bridge = points[168]
        
        left_brow_height = left_brow[1] - nose_bridge[1]
        right_brow_height = right_brow[1] - nose_bridge[1]
        metrics['brow_height'] = (left_brow_height + right_brow_height) / 2
        
        # Mouth corner metrics
        left_corner = points[61]
        right_corner = points[291]
        metrics['smile_ratio'] = (left_corner[1] + right_corner[1]) / 2
        
        return metrics

    def calculate_emotion_features(self, landmarks):
        """Calculate emotion-related features from facial landmarks"""
        features = {}
        
        # Convert landmarks to numpy array
        points = landmarks.astype(np.float32)
        
        # Mouth features
        mouth_points = points[48:68]
        mouth_width = np.linalg.norm(mouth_points[6] - mouth_points[0])
        mouth_height = np.linalg.norm(mouth_points[3] - mouth_points[9])
        features['mouth_ratio'] = mouth_height / mouth_width
        
        # Calculate mouth curvature (smile/frown)
        mouth_corners = (mouth_points[0] + mouth_points[6]) / 2
        mouth_center = mouth_points[3]
        features['mouth_curve'] = (mouth_corners[1] - mouth_center[1]) / mouth_width
        
        # Eye features
        left_eye = points[36:42]
        right_eye = points[42:48]
        
        # Eye openness
        left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5])
        right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5])
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3])
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3])
        
        features['eye_aspect_ratio'] = (left_eye_height/left_eye_width + 
                                      right_eye_height/right_eye_width) / 2
        
        # Eyebrow features
        left_brow = points[17:22]
        right_brow = points[22:27]
        nose_bridge = points[27:31]
        
        # Calculate eyebrow height relative to nose bridge
        left_brow_height = np.mean(left_brow[:, 1]) - np.mean(nose_bridge[:, 1])
        right_brow_height = np.mean(right_brow[:, 1]) - np.mean(nose_bridge[:, 1])
        features['brow_height'] = (left_brow_height + right_brow_height) / 2
        
        return features

    def analyze_face_regions(self, face_roi):
        """Analyze different regions of the face for emotions"""
        h, w = face_roi.shape[:2]
        features = {}
        
        # Define regions of interest
        eye_region_top = int(h * 0.2)
        eye_region_bottom = int(h * 0.5)
        mouth_region_top = int(h * 0.6)
        mouth_region_bottom = int(h * 0.9)
        
        # Extract regions
        eye_region = face_roi[eye_region_top:eye_region_bottom, :]
        mouth_region = face_roi[mouth_region_top:mouth_region_bottom, :]
        
        # Calculate features
        features['eye_intensity'] = np.mean(eye_region)
        features['eye_variance'] = np.var(eye_region)
        features['mouth_intensity'] = np.mean(mouth_region)
        features['mouth_variance'] = np.var(mouth_region)
        
        # Gradient features
        sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
        
        features['mouth_gradient_x'] = np.mean(sobelx[mouth_region_top:mouth_region_bottom, :])
        features['eye_gradient_y'] = np.mean(sobely[eye_region_top:eye_region_bottom, :])
        
        return features

    def detect_face_emotion(self, face_img):
        """Detect facial emotions using region analysis"""
        try:
            # Convert to grayscale and normalize
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                emotions = {
                    'happy': 0.0,
                    'sad': 0.0,
                    'angry': 0.0,
                    'neutral': 0.0,
                    'surprise': 0.0,
                    'fear': 0.0
                }
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (64, 64))
                    
                    # Get face region features
                    features = self.analyze_face_regions(face_roi)
                    
                    # Detect emotions based on features
                    # Happy - high mouth variance and positive mouth gradient
                    if features['mouth_variance'] > 1000 and features['mouth_gradient_x'] > 0:
                        emotions['happy'] = min(1.0, features['mouth_variance'] / 2000)
                    
                    # Sad - low mouth variance and negative mouth gradient
                    elif features['mouth_variance'] < 500 and features['mouth_gradient_x'] < 0:
                        emotions['sad'] = min(1.0, -features['mouth_gradient_x'] / 100)
                    
                    # Surprise - high eye variance and positive eye gradient
                    if features['eye_variance'] > 1500 and features['eye_gradient_y'] > 0:
                        emotions['surprise'] = min(1.0, features['eye_variance'] / 3000)
                    
                    # Angry - low eye variance and negative eye gradient
                    if features['eye_variance'] < 800 and features['eye_gradient_y'] < 0:
                        emotions['angry'] = min(1.0, -features['eye_gradient_y'] / 100)
                    
                    # Fear - very high eye variance
                    if features['eye_variance'] > 2000:
                        emotions['fear'] = min(1.0, features['eye_variance'] / 4000)
                    
                    # Set neutral if no strong emotions
                    if all(v < 0.2 for v in emotions.values()):
                        emotions['neutral'] = 0.7
                
                return emotions
            return None
        except Exception as e:
            print(f"Face emotion detection error: {str(e)}")
            return None

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to process audio chunks"""
        if status:
            print(f"Audio status: {status}")
            return
        
        try:
            # Convert to mono if stereo
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]
            
            # Ensure audio data is float32 and normalized
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Calculate RMS level
            current_level = np.sqrt(np.mean(audio_data**2))
            
            # Update noise floor estimation
            self.noise_history.append(current_level)
            if len(self.noise_history) > self.noise_window:
                self.noise_history.pop(0)
            
            # Estimate noise floor as the 10th percentile of recent levels
            noise_floor = np.percentile(self.noise_history, 10)
            
            # Subtract noise floor and ensure non-negative
            adjusted_level = max(0, current_level - noise_floor)
            
            # Apply threshold
            self.last_audio_level = adjusted_level if adjusted_level > self.noise_floor else 0
            
            # Process audio only if above noise floor
            if self.last_audio_level > self.noise_floor:
                # Apply noise reduction
                audio_data = audio_data - (noise_floor * np.sign(audio_data))
                audio_data = np.clip(audio_data, -1, 1)
                
                audio_features = self.audio_detector.extract_features_from_array(
                    audio_data, self.RATE)
                audio_emotions = self.audio_detector.detect_emotion(audio_features)
            else:
                audio_emotions = {'scream': 0.0, 'distress': 0.0}
            
            # Add audio level to emotions dict
            audio_emotions['audio_level'] = self.last_audio_level
            self.audio_queue.put(audio_emotions)
        except Exception as e:
            print(f"Audio processing error: {str(e)}")

    def start_audio_stream(self):
        """Start audio stream using sounddevice"""
        self.is_recording = True
        try:
            # Configure stream with proper settings
            self.audio_stream = sd.InputStream(
                channels=self.CHANNELS,
                samplerate=self.RATE,
                blocksize=self.BLOCKSIZE,
                dtype=np.float32,  # Specify data type
                device=None,  # Use default input device
                latency='low',  # Lower latency for real-time processing
                callback=self.audio_callback
            )
            print("Starting audio stream...")
            self.audio_stream.start()
            print("Audio stream started successfully")
        except Exception as e:
            print(f"Error starting audio stream: {str(e)}")
            print("Available audio devices:")
            print(sd.query_devices())
            self.is_recording = False

    def stop_audio_stream(self):
        """Stop the audio stream"""
        self.is_recording = False
        if hasattr(self, 'audio_stream'):
            print("Stopping audio stream...")
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                print("Audio stream stopped successfully")
            except Exception as e:
                print(f"Error stopping audio stream: {str(e)}")

    def draw_detections(self, frame, visual_detections, face_emotions):
        """Draw detections and emotions on frame"""
        # Get latest audio emotions
        audio_emotions = None
        try:
            while not self.audio_queue.empty():
                audio_emotions = self.audio_queue.get_nowait()
        except queue.Empty:
            pass

        # Draw audio level indicator
        if audio_emotions:
            audio_level = audio_emotions.get('audio_level', 0)
            # Draw audio level bar at top of frame
            bar_height = 20
            bar_width = int(frame.shape[1] * 0.3)
            x1 = 10
            y1 = 10
            # Background bar
            cv2.rectangle(frame, (x1, y1), (x1 + bar_width, y1 + bar_height), 
                         (50, 50, 50), -1)
            # Level bar
            # Only show levels above noise floor
            if audio_level > self.noise_floor:
                level_width = int(bar_width * min(1.0, audio_level / self.audio_threshold))
                # Color gradient from green to red based on level
                ratio = min(1.0, audio_level / self.audio_threshold)
                color = (0, int(255 * (1 - ratio)), int(255 * ratio))
            else:
                level_width = 0
                color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x1 + level_width, y1 + bar_height), 
                         color, -1)
            cv2.putText(frame, "Audio Level", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw audio emotions in top-right corner if available
        if audio_emotions:
            # Filter out low confidence emotions and audio_level key
            significant_emotions = {k: v for k, v in audio_emotions.items() 
                                     if v > 0.2 and k != 'audio_level'}
            
            y_pos = 30
            if significant_emotions:
                cv2.putText(frame, "Audio Emotions:", (frame.shape[1]-200, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 25
                for emotion, score in significant_emotions.items():
                    color = self.emotion_colors.get(emotion, (0, 165, 255))
                    text = f"{emotion}: {score:.0%}"
                    cv2.putText(frame, text, (frame.shape[1]-200, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_pos += 25

            # Draw audio alert if level is high
            if audio_emotions.get('audio_level', 0) > self.audio_threshold:
                cv2.putText(frame, "! HIGH AUDIO LEVEL !", 
                            (frame.shape[1]//2 - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for i, (x1, y1, x2, y2) in enumerate(visual_detections):
            # Draw person rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw emotions if available
            if i < len(face_emotions) and face_emotions[i]:
                emotions = face_emotions[i]
                y_offset = 25
                
                # Draw each emotion above the person
                for emotion, score in emotions.items():
                    if score > 0.2:  # Only show significant emotions
                        color = self.emotion_colors.get(emotion, (0, 255, 0))
                        # Draw emotion text
                        text = f'{emotion.capitalize()}: {score:.0%}'
                        cv2.putText(frame, text, (x1, y1-y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Draw emotion bar
                        bar_width = 50
                        bar_height = 4
                        filled_width = int(bar_width * score)
                        cv2.rectangle(frame, (x1+100, y1-y_offset-3), 
                                    (x1+100+bar_width, y1-y_offset+1), (0, 0, 0), 1)
                        cv2.rectangle(frame, (x1+100, y1-y_offset-3), 
                                    (x1+100+filled_width, y1-y_offset+1), color, -1)
                        y_offset += 20
            else:
                cv2.putText(frame, 'Person', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw audio distress warning above person if detected
        if audio_emotions and (audio_emotions.get('scream', 0) > 0.7 or 
                             audio_emotions.get('distress', 0) > 0.7):
            warning_text = "! AUDIO DISTRESS !"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, 2)[0]
            text_x = x1 + (x2-x1)//2 - text_size[0]//2
            cv2.putText(frame, warning_text, (text_x, y1-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def process_webcam(self, camera_id=0):
        """Process live webcam feed with audio"""
        # Initialize audio before video
        print("Initializing audio stream...")
        self.start_audio_stream()
        if not self.is_recording:
            print("Warning: Audio stream not started")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.stop_audio_stream()
            raise ValueError("Could not open webcam")

        print("Processing webcam feed... Press 'q' to quit")
        
        try:
            prev_detections = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process visual detection
                visual_detections = self.visual_detector.detect_people(frame)
                
                # Process face emotions for each detection
                face_emotions = []
                for i, detection in enumerate(visual_detections):
                    if len(detection) == 4:  # Ensure detection has x1,y1,x2,y2
                        x1, y1, x2, y2 = detection
                        face_height = int((y2 - y1) * 0.4)
                        margin_x = int((x2 - x1) * 0.1)
                        margin_y = int(face_height * 0.1)
                        face_x1 = max(0, x1 - margin_x)
                        face_y1 = max(0, y1 - margin_y)
                        face_x2 = min(frame.shape[1], x2 + margin_x)
                        face_y2 = min(frame.shape[0], y1 + face_height + margin_y)
                        face_img = frame[face_y1:face_y2, face_x1:face_x2]
                        
                        if face_img.size > 0:
                            emotions = self.detect_face_emotion(face_img)
                            if emotions:
                                # Apply temporal smoothing using person ID
                                person_id = i
                                smoothed_emotions = self.smooth_emotions(emotions, person_id)
                                face_emotions.append(smoothed_emotions)
                            else:
                                face_emotions.append({})
                        else:
                            face_emotions.append({})
                    else:
                        face_emotions.append({})

                # Draw detections and emotions
                frame = self.draw_detections(frame, visual_detections, face_emotions)
                
                # Display the frame
                cv2.imshow('Live Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            print("Cleaning up resources...")
            # Clean up
            self.stop_audio_stream()
            cap.release()
            cv2.destroyAllWindows() 