import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.model_selection import train_test_split

class LiveProcessor3:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.datasets_dir = os.path.join(self.base_dir, 'datasets')
        
        # Set Kaggle credentials path
        self.kaggle_json = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.kaggle_json), exist_ok=True)
        
        # Check and copy Kaggle credentials if needed
        self.setup_kaggle_credentials()
        
        # Download datasets and train models
        self.download_datasets()
        self.load_or_train_models()
        
        # Initialize detection components
        self.setup_components()
        
        # Labels and colors
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.injury_labels = ['normal', 'cut', 'bruise', 'burn', 'infection', 'rash']
        self.pose_labels = ['standing', 'sitting', 'lying', 'walking', 'running']
        
        self.emotion_colors = {
            'angry': (0, 0, 255),    # Red
            'happy': (0, 255, 0),    # Green
            'sad': (255, 0, 0),      # Blue
            'surprise': (0, 255, 255),# Yellow
            'fear': (255, 0, 255),   # Purple
            'disgust': (128, 0, 0),  # Dark Blue
            'neutral': (128, 128, 128)# Gray
        }

    def setup_kaggle_credentials(self):
        """Set up Kaggle credentials using the provided path"""
        try:
            # Check if credentials exist in the correct location
            if not os.path.exists(self.kaggle_json):
                # Create .kaggle directory if it doesn't exist
                os.makedirs(os.path.dirname(self.kaggle_json), exist_ok=True)
                
                # Copy credentials from source to .kaggle directory
                source_json = os.path.join(os.path.expanduser('~'), 'kaggle.json')
                if os.path.exists(source_json):
                    import shutil
                    shutil.copy2(source_json, self.kaggle_json)
                    # Set correct permissions
                    os.chmod(self.kaggle_json, 0o600)
                    print("Kaggle credentials set up successfully!")
                else:
                    raise FileNotFoundError("Kaggle credentials not found in source location")
            
            # Verify credentials are valid
            import json
            with open(self.kaggle_json, 'r') as f:
                creds = json.load(f)
                if 'username' not in creds or 'key' not in creds:
                    raise ValueError("Invalid Kaggle credentials format")
                
        except Exception as e:
            print(f"Error setting up Kaggle credentials: {str(e)}")
            print("Please ensure kaggle.json is properly configured")
            raise

    def download_datasets(self):
        """Download required datasets from Kaggle"""
        try:
            print("Initializing Kaggle API...")
            api = KaggleApi()
            api.authenticate()
            
            # Define datasets to download (using verified accessible datasets)
            datasets = {
                'emotions': {
                    'name': 'jonathanoheix/face-expression-recognition-dataset',
                    'path': os.path.join(self.datasets_dir, 'fer2013')
                },
                'injuries': {
                    'name': 'yasinpratomo/wound-dataset',  # Wound dataset
                    'path': os.path.join(self.datasets_dir, 'injuries')
                },
                'pose': {
                    'name': 'trainingdatapro/pose-estimation',  # Updated pose dataset
                    'path': os.path.join(self.datasets_dir, 'pose')
                }
            }
            
            # Download each dataset if not already present
            for dataset_name, info in datasets.items():
                if not os.path.exists(info['path']):
                    print(f"Downloading {dataset_name} dataset...")
                    try:
                        api.dataset_download_files(info['name'], path=info['path'], unzip=True)
                        print(f"{dataset_name} dataset downloaded successfully!")
                    except Exception as e:
                        print(f"Error downloading {dataset_name} dataset: {str(e)}")
                        # Try alternative datasets if primary fails
                        if dataset_name == 'pose':
                            alt_datasets = [
                                'trainingdatapro/pose-estimation',  # Primary pose dataset
                                'niharika41298/yoga-poses-dataset', # Backup dataset
                                'sakunaharinda/human-poses-dataset' # Second backup
                            ]
                            for alt_dataset in alt_datasets:
                                try:
                                    print(f"Trying alternative pose dataset: {alt_dataset}")
                                    api.dataset_download_files(alt_dataset, path=info['path'], unzip=True)
                                    print(f"Successfully downloaded alternative pose dataset!")
                                    break
                                except:
                                    continue
                else:
                    print(f"{dataset_name} dataset already exists.")
                    
        except Exception as e:
            print(f"Error downloading datasets: {str(e)}")
            print("Please ensure you have set up your Kaggle API credentials")

    def load_or_train_models(self):
        """Load or train emotion and injury models"""
        try:
            # Initialize models as None first
            self.emotion_model = None
            self.injury_model = None
            
            # Train emotion model
            print("Training emotion detection model...")
            self.train_emotion_model()
            
            # Train injury model
            print("Training injury detection model...")
            self.train_injury_model()

        except Exception as e:
            print(f"Error in model loading/training: {str(e)}")

    def setup_components(self):
        """Initialize detection components"""
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def train_emotion_model(self):
        """Train emotion model using local emotion folders"""
        try:
            # Get emotion dataset path
            emotion_path = os.path.join(self.datasets_dir, 'fer2013', 'train')
            if not os.path.exists(emotion_path):
                raise FileNotFoundError(f"Emotion dataset folder not found at {emotion_path}")
            
            # Get emotion types from subdirectories
            emotion_types = [d for d in os.listdir(emotion_path) 
                           if os.path.isdir(os.path.join(emotion_path, d))]
            
            if not emotion_types:
                raise ValueError(f"No emotion type folders found in {emotion_path}")
            
            print(f"Found emotion types: {emotion_types}")
            self.emotion_labels = emotion_types
            
            # Create data generator with augmentation
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            # Load training data
            print("Loading training data...")
            train_generator = datagen.flow_from_directory(
                emotion_path,
                target_size=(48, 48),
                color_mode='grayscale',
                batch_size=32,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            # Load validation data
            print("Loading validation data...")
            validation_generator = datagen.flow_from_directory(
                emotion_path,
                target_size=(48, 48),
                color_mode='grayscale',
                batch_size=32,
                class_mode='categorical',
                subset='validation',
                shuffle=True
            )
            
            print(f"Found {len(train_generator.class_indices)} emotion classes: {train_generator.class_indices}")
            
            # Create emotion model
            print("Creating emotion model...")
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(emotion_types), activation='softmax')
            ])
            
            # Compile model
            print("Compiling model...")
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Add callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3
                )
            ]
            
            # Train model
            print("Training emotion model...")
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=5,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model
            print("Saving emotion model...")
            model.save(os.path.join(self.models_dir, 'emotion_model.h5'))
            self.emotion_model = model
            
            print("Emotion model training completed!")
            
        except Exception as e:
            print(f"Error training emotion model: {str(e)}")
            print(f"Please ensure emotion dataset is properly organized in: {self.datasets_dir}/fer2013/train/")
            print("Expected structure:")
            print("fer2013/")
            print("  └── train/")
            print("      ├── angry/")
            print("      ├── happy/")
            print("      ├── sad/")
            print("      └── other_emotions/")

    def train_injury_model(self):
        """Train injury model using wound dataset folders"""
        try:
            # Get specific wound dataset path
            injury_path = os.path.join(self.datasets_dir, 'injuries', 'Wound_dataset')
            if not os.path.exists(injury_path):
                raise FileNotFoundError(f"Wound dataset folder not found at {injury_path}")
            
            # Get wound types from subdirectories
            wound_types = [d for d in os.listdir(injury_path) 
                         if os.path.isdir(os.path.join(injury_path, d))]
            
            if not wound_types:
                raise ValueError(f"No wound type folders found in {injury_path}")
            
            print(f"Found wound types: {wound_types}")
            self.injury_labels = wound_types
            
            # Create data generator with augmentation specific to wound images
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                zoom_range=0.2,
                fill_mode='nearest'
            )
            
            # Load training data
            print("Loading training data...")
            train_generator = datagen.flow_from_directory(
                injury_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            # Load validation data
            print("Loading validation data...")
            validation_generator = datagen.flow_from_directory(
                injury_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='validation',
                shuffle=True
            )
            
            print(f"Found {len(train_generator.class_indices)} classes: {train_generator.class_indices}")
            
            # Create model using EfficientNetB0
            print("Creating model...")
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(len(wound_types), activation='softmax')
            ])
            
            # Compile model
            print("Compiling model...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'AUC', 'Precision', 'Recall']
            )
            
            # Add callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3
                )
            ]
            
            # Train model
            print("Training model...")
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=5,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model
            print("Saving model...")
            model.save(os.path.join(self.models_dir, 'injury_model.h5'))
            self.injury_model = model
            
            print("Injury model training completed!")
            
        except Exception as e:
            print(f"Error training injury model: {str(e)}")
            print(f"Please ensure wound dataset is properly organized in: {self.datasets_dir}/injuries/Wound_dataset/")
            print("Expected structure:")
            print("injuries/")
            print("  └── Wound_dataset/")
            print("      ├── wound_type_1/")
            print("      │   ├── image1.jpg")
            print("      │   └── image2.jpg")
            print("      ├── wound_type_2/")
            print("      └── wound_type_3/")

    def detect_emotions(self, face_roi):
        """Detect emotions in face ROI"""
        try:
            # Preprocess image
            face = cv2.resize(face_roi, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            
            # Get predictions
            predictions = self.emotion_model.predict(face)
            
            # Convert to dictionary
            emotions = {}
            for i, label in enumerate(self.emotion_labels):
                emotions[label] = float(predictions[0][i] * 100)
            
            return emotions
            
        except Exception as e:
            print(f"Error detecting emotions: {str(e)}")
            return None

    def detect_injuries(self, roi):
        """Detect injuries in region of interest"""
        try:
            # Preprocess image
            img = cv2.resize(roi, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Get predictions
            predictions = self.injury_model.predict(img)
            
            # Convert to dictionary
            injuries = {}
            for i, label in enumerate(self.injury_labels):
                score = float(predictions[0][i] * 100)
                if score > 20:  # Only include if confidence > 20%
                    injuries[label] = score
            
            return injuries
            
        except Exception as e:
            print(f"Error detecting injuries: {str(e)}")
            return None

    def process_webcam(self, camera_id=0):
        """Process webcam feed with trained models and detailed landmarks"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not cap.isOpened():
            raise ValueError("Could not open webcam")

        print("Processing webcam feed... Press 'q' to quit")
        
        frame_count = 0
        last_emotions = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_count += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process face mesh and detections
                results = self.face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Get face bounding box
                        h, w, _ = frame.shape
                        x_min = w
                        x_max = 0
                        y_min = h
                        y_max = 0
                        
                        for landmark in face_landmarks.landmark:
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            x_min = min(x_min, x)
                            x_max = max(x_max, x)
                            y_min = min(y_min, y)
                            y_max = max(y_max, y)
                        
                        # Draw face rectangle
                        x1, y1, x2, y2 = max(0, x_min-10), max(0, y_min-10), min(w, x_max+10), min(h, y_max+10)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Extract face ROI
                        face_roi = frame[y1:y2, x1:x2]
                        
                        if face_roi.size > 0:
                            # Draw detailed facial landmarks
                            self.draw_detailed_facial_features(frame, face_landmarks, face_roi, x1, y1, x2, y2)
                            
                            # Process emotions every 3rd frame
                            if frame_count % 3 == 0:
                                emotions = self.detect_emotions(face_roi)
                                if emotions:
                                    last_emotions = emotions
                                    # Draw emotion labels
                                    y_offset = y1 - 10
                                    for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]:
                                        if score > 20:
                                            label = f"{emotion.capitalize()} ({score:.0f}%)"
                                            color = self.emotion_colors.get(emotion, (255, 255, 255))
                                            cv2.putText(frame, label, (x1, y_offset),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                            y_offset -= 25
                            
                            # Process injuries
                            injuries = self.detect_injuries(face_roi)
                            if injuries:
                                y_offset = y2 + 25
                                for injury, score in injuries.items():
                                    if score > 20:
                                        label = f"{injury}: {score:.0f}%"
                                        cv2.putText(frame, label, (x1, y_offset),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        y_offset += 25
                
                cv2.imshow('Biometric Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def draw_detailed_facial_features(self, frame, landmarks, face_roi, x1, y1, x2, y2):
        """Draw detailed facial features like LiveProcessor2"""
        face_width = x2 - x1
        face_height = y2 - y1
        
        def get_point(landmark):
            return (
                int(landmark.x * face_width + x1),
                int(landmark.y * face_height + y1)
            )
        
        # Draw eyebrows
        left_eyebrow = [get_point(landmarks.landmark[p]) for p in [70, 63, 105, 66, 107]]
        right_eyebrow = [get_point(landmarks.landmark[p]) for p in [336, 296, 334, 293, 300]]
        
        for i in range(len(left_eyebrow) - 1):
            cv2.line(frame, left_eyebrow[i], left_eyebrow[i + 1], (0, 255, 0), 1)
            cv2.line(frame, right_eyebrow[i], right_eyebrow[i + 1], (0, 255, 0), 1)
        
        # Draw eyes
        left_eye = [get_point(landmarks.landmark[p]) for p in [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]]
        right_eye = [get_point(landmarks.landmark[p]) for p in [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]]
        
        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
        
        # Draw mouth
        outer_lips = [get_point(landmarks.landmark[p]) for p in [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]]
        inner_lips = [get_point(landmarks.landmark[p]) for p in [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]]
        
        cv2.polylines(frame, [np.array(outer_lips)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(inner_lips)], True, (0, 255, 0), 1)