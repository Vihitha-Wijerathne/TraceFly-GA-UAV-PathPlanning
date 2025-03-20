import cv2
import numpy as np
import mediapipe as mp
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
from collections import deque
from deepface import DeepFace
import tensorflow as tf
import time
import os
import h5py
from fastai.vision.all import load_learner
from PIL import Image
import pickle
import traceback
import torch
import scipy.ndimage as ndimage
import scipy.fftpack as fftpack

class LiveProcessor2:
    def __init__(self):
        self.visual_detector = VisualDetector()
        self.audio_detector = AudioDetector()
        
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
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
            'fear': (0, 165, 255),      # Orange
            'disgust': (128, 0, 128)   # Purple
        }
        
        # rPPG parameters for heart rate detection
        self.ppg_buffer_size = 300  # 10 seconds at 30 fps
        self.ppg_buffer = deque(maxlen=self.ppg_buffer_size)
        self.last_hr_time = datetime.now()
        self.hr_update_interval = 3  # Update heart rate every 3 seconds
        self.current_hr = 0

        # Add pose classifications
        self.pose_classifications = {
            'standing': ['standing_still', 'standing_active'],
            'sitting': ['sitting_straight', 'sitting_relaxed', 'sitting_forward'],
            'lying': ['lying_back', 'lying_side'],
            'walking': ['walking_slow', 'walking_fast'],
            'running': ['running'],
            'jumping': ['jumping'],
            'crouching': ['crouching'],
            'hands_raised': ['hands_raised'],
            'hands_waving': ['waving']
        }
        
        # Add pose colors
        self.pose_colors = {
            'standing': (0, 255, 0),    # Green
            'sitting': (255, 255, 0),   # Yellow
            'lying': (0, 165, 255),     # Orange
            'walking': (255, 0, 255)    # Purple
        }

        # Use DeepFace directly for emotion detection
        self.use_deepface = True
        
        # Initialize pose estimation with BlazePose (more accurate than basic MediaPipe)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=2,  # Use the most accurate model
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            enable_segmentation=True
        )

        # Initialize emotion detection
        try:
            # Try to load the FER model with error handling
            model_path = 'models/fer_model.h5'
            if not Path(model_path).exists():
                print(f"Warning: FER model not found at {model_path}")
                self.fer_model = None
            else:
                self.fer_model = tf.keras.models.load_model(model_path, compile=False)
                # Remove the learning rate argument that's causing issues
                if self.fer_model:
                    self.fer_model.compile(optimizer='adam')
        except Exception as e:
            print(f"Error loading FER model: {str(e)}")
            self.fer_model = None
        
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Optimize for Intel Iris Xe
        self.use_gpu = False
        try:
            # Check if we can use OpenCL acceleration
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                print("OpenCL acceleration enabled")
                self.use_gpu = True
        except:
            print("OpenCL acceleration not available")

        # Initialize MediaPipe with lighter settings for integrated GPU
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Use lighter model settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Disable for better performance
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3  # Lower for better performance
        )
        
        self.pose = self.mp_pose.Pose(
            model_complexity=1,  # Use medium complexity model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            enable_segmentation=False  # Disable for better performance
        )

        # Add injury detection parameters
        self.categories = ('cuts_and_wounds', 'fracture', 'rash', 'splinter')
        self.injury_types = {
            'cuts_and_wounds': (0, 0, 255),    # Red
            'fracture': (128, 0, 128),         # Purple
            'rash': (0, 255, 255),             # Yellow
            'splinter': (255, 128, 0)          # Orange
        }

        # Initialize paths with Windows-compatible paths
        try:
            # Use os.path for Windows compatibility
            self.models_dir = os.path.abspath(os.path.join(os.getcwd(), 'models'))
            print(f"Looking for models in: {self.models_dir}")
            
            dataset_path = os.path.join(self.models_dir, 'wound_dataset.h5')
            print(f"Loading dataset from: {dataset_path}")
            
            if not os.path.exists(self.models_dir):
                print(f"Models directory does not exist: {self.models_dir}")
                os.makedirs(self.models_dir, exist_ok=True)
                print("Created models directory")
            
            if os.path.exists(dataset_path):
                print(f"Found dataset file at: {dataset_path}")
                try:
                    with h5py.File(dataset_path, 'r') as f:
                        # Load categories from label mapping
                        if 'label_mapping' in f:
                            self.categories = []
                            for key, value in f['label_mapping'].attrs.items():
                                self.categories.append(key)
                            print(f"Loaded categories: {self.categories}")
                            
                            # Update injury types with loaded categories
                            self.injury_types = {
                                str(cat): (0, 0, 255) for cat in self.categories
                            }
                            
                            # Load the model weights and data
                            self.images = f['images'][:]
                            self.labels = f['labels'][:]
                            print(f"Loaded {len(self.images)} training images")
                            
                            # Create a simple model for inference
                            self.injury_model = tf.keras.Sequential([
                                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
                                tf.keras.layers.MaxPooling2D(),
                                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                                tf.keras.layers.MaxPooling2D(),
                                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(64, activation='relu'),
                                tf.keras.layers.Dense(len(self.categories), activation='softmax')
                            ])
                            
                            # Train the model on the loaded data
                            self.injury_model.compile(
                                optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy']
                            )
                            
                            print("Training model on loaded data...")
                            self.injury_model.fit(
                                self.images, 
                                self.labels,
                                epochs=5,
                                batch_size=32,
                                validation_split=0.2
                            )
                            
                            print("Successfully loaded and trained injury detection model")
                except Exception as load_error:
                    print(f"Error loading dataset file: {str(load_error)}")
                    print("Detailed error:", traceback.format_exc())
                    self.injury_model = None
            else:
                print(f"No dataset file found at: {dataset_path}")
                print("Please ensure 'wound_dataset.h5' is in the models directory")
                self.injury_model = None
        except Exception as e:
            print(f"Error in model initialization: {str(e)}")
            print("Detailed error:", traceback.format_exc())
            self.injury_model = None

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

    def extract_face_landmarks(self, frame):
        """Extract facial landmarks using MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        landmarks = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks.append(face_landmarks)
        
        return landmarks

    def detect_pose(self, frame):
        """Detect body pose using MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results.pose_landmarks

    def estimate_heart_rate(self, frame, face_landmarks):
        """Estimate heart rate using rPPG"""
        if not face_landmarks:
            return
        
        # Define ROI for face (cheeks and forehead)
        h, w = frame.shape[:2]
        roi_points = []
        for landmark in face_landmarks[0].landmark:
            roi_points.append([int(landmark.x * w), int(landmark.y * h)])
        
        roi_points = np.array(roi_points)
        
        # Create mask for ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, roi_points, 1)
        
        # Extract average RGB values from ROI
        rgb_means = cv2.mean(frame, mask=mask)[:3]
        self.ppg_buffer.append(rgb_means)
        
        # Calculate heart rate every few seconds
        if (datetime.now() - self.last_hr_time).seconds >= self.hr_update_interval:
            if len(self.ppg_buffer) >= self.ppg_buffer_size:
                self.current_hr = self.calculate_heart_rate()
                self.last_hr_time = datetime.now()
        
        return self.current_hr

    def calculate_heart_rate(self):
        """Calculate heart rate from PPG buffer"""
        # Convert buffer to numpy array
        signals = np.array(self.ppg_buffer)
        
        # Normalize signals
        normalized = signals - np.mean(signals, axis=0)
        
        # Apply bandpass filter (0.7-4Hz for heart rate 42-240 BPM)
        fps = 30
        nyquist = fps/2
        low = 0.7/nyquist
        high = 4.0/nyquist
        b, a = butter(3, [low, high], btype='band')
        filtered = filtfilt(b, a, normalized, axis=0)
        
        # Get green channel and perform FFT
        green_signal = filtered[:, 1]  # Green channel
        fft = np.fft.fft(green_signal)
        frequencies = np.fft.fftfreq(len(green_signal), d=1/fps)
        
        # Find dominant frequency in expected HR range (42-240 BPM)
        freq_mask = (frequencies >= 0.7) & (frequencies <= 4.0)
        peaks = np.abs(fft)[freq_mask]
        peak_freq = frequencies[freq_mask][np.argmax(peaks)]
        
        # Convert frequency to BPM
        hr = peak_freq * 60
        
        return int(hr)

    def analyze_emotions_from_landmarks(self, landmarks, image_shape):
        """Analyze emotions based on facial landmarks and pose"""
        emotions = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'neutral': 0.0,
            'surprise': 0.0,
            'fear': 0.0
        }
        
        if not landmarks:
            return emotions
            
        # Extract facial metrics
        metrics = self.calculate_face_metrics(landmarks[0], image_shape)
        
        # Analyze emotions based on facial metrics
        # Happy - raised cheeks, wide mouth
        if metrics['mouth_ratio'] > 0.5 and metrics['smile_ratio'] < 0.4:
            emotions['happy'] = min(1.0, metrics['mouth_ratio'])
            
        # Sad - drooping mouth corners, lowered brows
        elif metrics['mouth_ratio'] < 0.3 and metrics['brow_height'] < -0.1:
            emotions['sad'] = min(1.0, abs(metrics['brow_height']))
            
        # Surprise - raised brows, wide eyes, open mouth
        if metrics['eye_openness'] > 0.35 and metrics['mouth_ratio'] > 0.6:
            emotions['surprise'] = min(1.0, metrics['eye_openness'])
            
        # Angry - lowered brows, tight mouth
        if metrics['brow_height'] < -0.15 and metrics['mouth_ratio'] < 0.25:
            emotions['angry'] = min(1.0, abs(metrics['brow_height']))
            
        # Fear - wide eyes, raised brows
        if metrics['eye_openness'] > 0.4 and metrics['brow_height'] > 0.2:
            emotions['fear'] = min(1.0, metrics['eye_openness'])
            
        # Neutral - balanced metrics
        if all(v < 0.3 for v in emotions.values()):
            emotions['neutral'] = 0.7
            
        return emotions

    def detect_emotions(self, frame):
        """
        Enhanced emotion detection using DeepFace with better accuracy
        """
        try:
            if frame is None or frame.size == 0:
                return None

            # Enhance image quality for better detection
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=30)
            
            try:
                # Use DeepFace with multiple models for better accuracy
                result1 = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                result2 = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='ssd'
                )
                
                if result1 and result2:
                    # Combine results from both models
                    emotions1 = result1[0]['emotion']
                    emotions2 = result2[0]['emotion']
                    
                    # Average the emotions
                    emotions = {}
                    for emotion in set(emotions1.keys()) | set(emotions2.keys()):
                        emotions[emotion] = (emotions1.get(emotion, 0) + emotions2.get(emotion, 0)) / 2
                    
                    # Filter low confidence emotions
                    emotions = {k: v for k, v in emotions.items() if v > 20}
                    
                    # Get top 2 emotions
                    top_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2])
                    return top_emotions

            except Exception as e:
                print(f"DeepFace error: {str(e)}")
                return {'neutral': 100.0}

            return {'neutral': 100.0}

        except Exception as e:
            print(f"Emotion detection error: {str(e)}")
            return {'neutral': 100.0}

    def classify_pose(self, pose_landmarks):
        """
        Enhanced pose classification using advanced geometric analysis
        """
        if not pose_landmarks:
            return None
            
        # Convert landmarks to numpy array
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])
        
        # Extract key points
        nose = landmarks[0]
        shoulders = landmarks[11:13]
        hips = landmarks[23:25]
        knees = landmarks[25:27]
        ankles = landmarks[27:29]
        wrists = landmarks[15:17]
        
        # Calculate key metrics
        shoulder_height = np.mean(shoulders[:, 1])
        hip_height = np.mean(hips[:, 1])
        knee_height = np.mean(knees[:, 1])
        ankle_height = np.mean(ankles[:, 1])
        
        # Vertical alignment
        spine_vertical = abs(np.mean(shoulders[:, 0]) - np.mean(hips[:, 0]))
        
        # Movement detection
        shoulder_movement = abs(shoulders[0][0] - shoulders[1][0])
        
        # Detailed pose classification
        if hip_height < 0.7:  # Standing poses
            if shoulder_movement > 0.2:
                if np.max(wrists[:, 1]) < shoulder_height:  # Hands raised
                    return 'hands_raised'
                elif abs(wrists[0][0] - wrists[1][0]) > 0.3:  # Hands waving
                    return 'hands_waving'
                return 'walking_fast' if shoulder_movement > 0.3 else 'walking_slow'
            return 'standing_active' if spine_vertical > 0.1 else 'standing_still'
            
        elif 0.7 <= hip_height < 0.8:  # Sitting poses
            if spine_vertical < 0.05:
                return 'sitting_straight'
            elif nose[1] > shoulder_height:
                return 'sitting_forward'
            return 'sitting_relaxed'
            
        elif knee_height < 0.3:  # Low poses
            return 'crouching'
            
        else:  # Lying poses
            return 'lying_back' if abs(shoulder_height - hip_height) < 0.1 else 'lying_side'

    def draw_facial_emotions(self, frame, face_roi, x1, y1, x2, y2):
        """Draw facial emotions following actual facial landmarks"""
        try:
            # Get face landmarks using MediaPipe
            face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Process the face ROI
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(face_roi_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Convert relative coordinates to absolute image coordinates
                face_width = x2 - x1
                face_height = y2 - y1
                
                def get_point(landmark):
                    return (
                        int(landmark.x * face_width + x1),
                        int(landmark.y * face_height + y1)
                    )
                
                # Draw facial landmarks
                # Draw eyebrows
                left_eyebrow = [get_point(landmarks[p]) for p in [70, 63, 105, 66, 107]]
                right_eyebrow = [get_point(landmarks[p]) for p in [336, 296, 334, 293, 300]]
                
                for i in range(len(left_eyebrow) - 1):
                    cv2.line(frame, left_eyebrow[i], left_eyebrow[i + 1], (0, 255, 0), 1)
                    cv2.line(frame, right_eyebrow[i], right_eyebrow[i + 1], (0, 255, 0), 1)
                
                # Draw eyes
                left_eye = [get_point(landmarks[p]) for p in [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]]
                right_eye = [get_point(landmarks[p]) for p in [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]]
                
                cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
                
                # Draw mouth
                outer_lips = [get_point(landmarks[p]) for p in [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]]
                inner_lips = [get_point(landmarks[p]) for p in [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]]
                
                cv2.polylines(frame, [np.array(outer_lips)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(inner_lips)], True, (0, 255, 0), 1)
                
                # Get emotion predictions but don't draw the label here
                if self.fer_model is not None:
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    face_roi_fer = cv2.resize(gray_roi, (48, 48))
                    face_roi_fer = face_roi_fer.astype('float32') / 255.0
                    face_roi_fer = np.expand_dims(face_roi_fer, axis=-1)
                    face_roi_fer = np.expand_dims(face_roi_fer, axis=0)
                    
                    predictions = self.fer_model.predict(face_roi_fer, verbose=0)
                    emotions = {self.emotion_labels[i]: float(score * 100) 
                              for i, score in enumerate(predictions[0])}
                    return emotions
                else:
                    result = DeepFace.analyze(
                        face_roi,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    return result[0]['emotion'] if result else None
            
            return None
            
        except Exception as e:
            print(f"Error in draw_facial_emotions: {str(e)}")
            return None

    def draw_detections(self, frame, visual_detections, face_emotions):
        """Draw detections with enhanced visualization"""
        for i, (x1, y1, x2, y2) in enumerate(visual_detections):
            # Draw person rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get face region
            face_roi = frame[y1:y2, x1:x2]
            
            # Draw facial emotions
            emotions = self.draw_facial_emotions(frame, face_roi, x1, y1, x2, y2)
            
            # Detect and draw pose
            pose_landmarks = self.detect_pose(frame)
            pose = self.classify_pose(pose_landmarks) if pose_landmarks else None
            
            if pose:
                # Draw pose label
                pose_y = y1 - 60  # Position above emotion labels
                cv2.putText(frame,
                           f"Pose: {pose.replace('_', ' ').title()}",
                           (x1, pose_y),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
            
            # Draw pose skeleton
            if pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 0, 0), thickness=3, circle_radius=3),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2)
                )
        
        return frame

    def get_emotion_color(self, emotion):
        """Get color for each emotion"""
        colors = {
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'neutral': (0, 255, 0),    # Green
            'surprise': (255, 0, 255),  # Purple
            'fear': (0, 165, 255),     # Orange
            'disgust': (128, 0, 128)   # Purple
        }
        return colors.get(emotion, (200, 200, 200))  # Gray for unknown emotions
    
    def detect_injuries(self, frame):
        """
        Simplified and more reliable injury detection
        """
        try:
            injuries = []
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define injury detection parameters
            injury_params = {
                'cuts_and_wounds': {
                    'lower': np.array([0, 100, 100]),  # Red color range
                    'upper': np.array([10, 255, 255]),
                    'min_area': 100,
                    'color': (0, 0, 255)  # Red
                },
                'rash': {
                    'lower': np.array([0, 50, 50]),    # Pink/Red range
                    'upper': np.array([20, 255, 255]),
                    'min_area': 150,
                    'color': (0, 255, 255)  # Yellow
                },
                'bruise': {
                    'lower': np.array([100, 50, 50]),  # Purple/Blue range
                    'upper': np.array([140, 255, 255]),
                    'min_area': 100,
                    'color': (128, 0, 128)  # Purple
                }
            }
            
            # Process each injury type
            for injury_type, params in injury_params.items():
                # Create mask for color range
                mask = cv2.inRange(hsv, params['lower'], params['upper'])
                
                # Noise reduction
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > params['min_area']:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Calculate confidence based on area
                        confidence = min((area / 1000) * 100, 100)
                        
                        # Add detection if confidence is high enough
                        if confidence > 30:
                            injuries.append({
                                'type': injury_type,
                                'location': (x, y, w, h),
                                'confidence': confidence,
                                'color': params['color']
                            })
            
            return injuries
            
        except Exception as e:
            print(f"Error in injury detection: {str(e)}")
            print("Detailed error:", traceback.format_exc())
            return []

    def draw_injuries(self, frame, injuries):
        """Draw detected injuries with simplified visualization"""
        try:
            if not injuries:
                return
                
            for injury in injuries:
                x, y, w, h = injury['location']
                injury_type = injury['type']
                confidence = injury['confidence']
                color = injury['color']
                
                # Draw rectangle around injury
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{injury_type}: {confidence:.0f}%"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        except Exception as e:
            print(f"Error drawing injuries: {str(e)}")

    def process_webcam(self, camera_id=0):
        """Process webcam feed with optimized performance"""
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")

        # Optimize camera settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        print("Processing webcam feed... Press 'q' to quit")
        
        # Create window
        cv2.namedWindow('Live Detection', cv2.WINDOW_NORMAL)
        
        # Initialize processing variables
        frame_count = 0
        last_emotions = None
        last_pose = None
        last_pose_landmarks = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_count += 1

                # Process visual detection
                visual_detections = self.visual_detector.detect_people(frame)
                
                # Process pose every 5th frame
                if frame_count % 5 == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(frame_rgb)
                    if pose_results.pose_landmarks:
                        last_pose_landmarks = pose_results.pose_landmarks
                        last_pose = self.classify_pose(last_pose_landmarks)
                
                for i, (x1, y1, x2, y2) in enumerate(visual_detections):
                    # Basic checks
                    if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                        continue

                    # Draw person rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Process face every 3rd frame
                    if frame_count % 3 == 0:
                        face_roi = frame[y1:y2, x1:x2]
                        if face_roi.size > 0:  # Check if ROI is valid
                            emotions = self.detect_emotions(face_roi)
                            if emotions:
                                last_emotions = emotions

                    # Add injury detection
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        injuries = self.detect_injuries(face_roi)
                        if injuries:
                            # Adjust injury coordinates to global frame
                            for injury in injuries:
                                ix, iy, iw, ih = injury['location']
                                injury['location'] = (ix + x1, iy + y1, iw, ih)
                            self.draw_injuries(frame, injuries)
                    
                    # Draw labels
                    label_parts = [f"Person {i+1}"]
                    
                    # Add emotions from last detection
                    if last_emotions:
                        for emotion, score in last_emotions.items():
                            if score > 20:
                                label_parts.append(f"{emotion.capitalize()} ({score:.0f}%)")
                    
                    # Add pose from last detection
                    if last_pose:
                        label_parts.append(last_pose.replace('_', ' ').title())
                    
                    # Draw labels efficiently
                    y_offset = y1 - 10
                    for j, part in enumerate(label_parts):
                        color = (0, 255, 0)
                        if j > 0 and j < len(label_parts) - 1:
                            emotion = part.split()[0].lower()
                            color = self.get_emotion_color(emotion)
                        elif j == len(label_parts) - 1 and last_pose:
                            color = self.pose_colors.get(last_pose.split('_')[0], (0, 255, 0))
                        
                        cv2.putText(frame, part, (x1, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_offset -= 25
                    
                    # Draw pose skeleton if available
                    if last_pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            last_pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                                color=(255, 0, 0), thickness=2, circle_radius=2),
                                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                                    color=(0, 255, 0), thickness=1)
                        )

                # Display frame
                cv2.imshow('Live Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows() 