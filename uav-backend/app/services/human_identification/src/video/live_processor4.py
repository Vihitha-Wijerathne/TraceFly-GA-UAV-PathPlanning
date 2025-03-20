import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import warnings
from datetime import datetime
import urllib.request
import ssl
import certifi
import sounddevice as sd
import threading
import queue
from collections import deque
import tensorflow as tf
import time
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import requests
import librosa
import matplotlib.pyplot as plt
import io
import base64
from scipy.io import wavfile
import pandas as pd

class AudioDistressProcessor:
    def __init__(self):
        # Initialize paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.datasets_dir = os.path.join(self.base_dir, 'datasets')
        self.audio_dir = os.path.join(self.datasets_dir, 'audio_distress')
        
        # Set Kaggle credentials path
        self.kaggle_json = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Audio distress labels
        self.distress_labels = ['crying', 'screaming', 'shouting', 'gasping', 'choking', 'wheezing', 'normal']
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.duration = 3  # seconds
        self.audio_buffer = deque(maxlen=self.sample_rate * self.duration)
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Load or download audio distress dataset and model
        self.download_dataset()
        self.load_or_train_model()
        
        # Start audio capture thread
        self.is_capturing = False
        self.audio_thread = None

    def download_dataset(self):
        """Download audio distress dataset from Kaggle"""
        try:
            # Check if dataset already exists
            if os.path.exists(self.audio_dir) and len(os.listdir(self.audio_dir)) > 0:
                print("Audio distress dataset already exists.")
                return
            
            print("Downloading audio distress dataset from Kaggle...")
            
            # Check if kaggle.json exists
            if not os.path.exists(self.kaggle_json):
                # Create .kaggle directory if it doesn't exist
                os.makedirs(os.path.dirname(self.kaggle_json), exist_ok=True)
                
                # Create a basic kaggle.json file with empty credentials
                # User will need to fill this with their actual credentials
                with open(self.kaggle_json, 'w') as f:
                    f.write('{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}')
                
                print(f"Created empty Kaggle credentials file at {self.kaggle_json}")
                print("Please edit this file with your Kaggle username and API key.")
                print("You can get your API key from https://www.kaggle.com/account")
                
                # Set permissions
                os.chmod(self.kaggle_json, 0o600)
                return
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset (using the specified dataset)
            dataset_name = "eliasmarcon/environmental-sound-classification-50"
            api.dataset_download_files(dataset_name, path=self.audio_dir)
            
            # Extract the dataset
            zip_path = os.path.join(self.audio_dir, 'environmental-sound-classification-50.zip')
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.audio_dir)
                os.remove(zip_path)
            
            # Create distress category folders
            for label in self.distress_labels:
                os.makedirs(os.path.join(self.audio_dir, label), exist_ok=True)
            
            # Map ESC-50 categories to our distress categories
            category_mapping = {
                # Distress sounds
                'crying_baby': 'crying',
                'crying_sobbing': 'crying',
                'screaming': 'screaming',
                'children_shouting': 'shouting',
                'shouting': 'shouting',
                'breathing': 'gasping',
                'gasping': 'gasping',
                'coughing': 'choking',
                'sneezing': 'wheezing',
                'wheezing': 'wheezing',
                
                # Normal sounds
                'footsteps': 'normal',
                'laughing': 'normal',
                'brushing_teeth': 'normal',
                'snoring': 'normal',
                'drinking_sipping': 'normal',
                'water_drops': 'normal',
                'clock_tick': 'normal',
                'clock_alarm': 'normal',
                'keyboard_typing': 'normal',
                'door_wood_knock': 'normal',
                'mouse_click': 'normal',
                'finger_snapping': 'normal',
                'clapping': 'normal',
                'speech': 'normal',
                'conversation': 'normal',
            }
            
            # Find and organize audio files
            for root, dirs, files in os.walk(self.audio_dir):
                for file in files:
                    if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.ogg'):
                        # Try to determine category from filename or path
                        file_path = os.path.join(root, file)
                        file_lower = file.lower()
                        path_lower = root.lower()
                        
                        # Find matching category
                        matched_category = None
                        for category, target in category_mapping.items():
                            if category in file_lower or category in path_lower:
                                matched_category = target
                                break
                        
                        # If no match found but contains distress keywords
                        if matched_category is None:
                            distress_keywords = ['scream', 'shout', 'cry', 'sob', 'gasp', 'chok', 'wheez']
                            for keyword in distress_keywords:
                                if keyword in file_lower or keyword in path_lower:
                                    if keyword in ['scream']:
                                        matched_category = 'screaming'
                                    elif keyword in ['shout']:
                                        matched_category = 'shouting'
                                    elif keyword in ['cry', 'sob']:
                                        matched_category = 'crying'
                                    elif keyword in ['gasp']:
                                        matched_category = 'gasping'
                                    elif keyword in ['chok']:
                                        matched_category = 'choking'
                                    elif keyword in ['wheez']:
                                        matched_category = 'wheezing'
                                    break
                        
                        # Default to normal if still no match
                        if matched_category is None:
                            matched_category = 'normal'
                        
                        # Copy file to appropriate category folder
                        target_dir = os.path.join(self.audio_dir, matched_category)
                        target_path = os.path.join(target_dir, file)
                        
                        # Only copy if source and target are different
                        if file_path != target_path:
                            import shutil
                            try:
                                shutil.copy2(file_path, target_path)
                                print(f"Copied {file} to {matched_category} category")
                            except Exception as e:
                                print(f"Error copying {file}: {str(e)}")
            
            print("Audio distress dataset downloaded and organized successfully.")
            
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            print("Please ensure you have valid Kaggle credentials in ~/.kaggle/kaggle.json")
            print("You can download datasets manually and place them in the appropriate folders.")

    def load_or_train_model(self):
        """Load existing audio distress model or train a new one"""
        model_path = os.path.join(self.models_dir, 'audio_distress_model.h5')
        
        try:
            if os.path.exists(model_path):
                print("Loading existing audio distress model...")
                self.model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully.")
            else:
                print("Training new audio distress model...")
                self.train_audio_model()
        except Exception as e:
            print(f"Error loading/training model: {str(e)}")
            self.model = None

    def extract_features(self, audio_data, sample_rate):
        """Extract audio features (MFCC, spectral centroid, etc.)"""
        try:
            # Ensure audio is the right length
            if len(audio_data) < sample_rate * self.duration:
                # Pad with zeros if too short
                padding = np.zeros(sample_rate * self.duration - len(audio_data))
                audio_data = np.concatenate([audio_data, padding])
            elif len(audio_data) > sample_rate * self.duration:
                # Truncate if too long
                audio_data = audio_data[:sample_rate * self.duration]
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            
            # Extract spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            
            # Extract spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            
            # Extract spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
            
            # Extract zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # Combine features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.mean(spectral_contrast, axis=1),
                np.std(spectral_contrast, axis=1),
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate)
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return np.zeros(50)  # Return empty features on error

    def train_audio_model(self):
        """Train audio distress detection model"""
        try:
            # Check if we have data for each category
            valid_categories = []
            features_list = []
            labels_list = []
            
            for i, category in enumerate(self.distress_labels):
                category_dir = os.path.join(self.audio_dir, category)
                if os.path.exists(category_dir):
                    audio_files = [f for f in os.listdir(category_dir) 
                                 if f.endswith('.wav') or f.endswith('.mp3') or f.endswith('.ogg')]
                    
                    if audio_files:
                        valid_categories.append(category)
                        print(f"Processing {len(audio_files)} files for category '{category}'...")
                        
                        # Process each audio file
                        for audio_file in audio_files[:100]:  # Limit to 100 files per category
                            file_path = os.path.join(category_dir, audio_file)
                            
                            try:
                                # Load audio file
                                audio_data, sample_rate = librosa.load(file_path, sr=self.sample_rate)
                                
                                # Extract features
                                features = self.extract_features(audio_data, sample_rate)
                                
                                # Add to dataset
                                features_list.append(features)
                                labels_list.append(i)
                            except Exception as e:
                                print(f"Error processing file {audio_file}: {str(e)}")
            
            if not features_list:
                raise ValueError("No valid audio files found for training")
            
            # Convert to numpy arrays
            X = np.array(features_list)
            y = np.array(labels_list)
            
            # Create one-hot encoded labels
            y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(valid_categories))
            
            # Split into training and validation sets
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
            
            # Create model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(len(valid_categories), activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )
            
            # Save model
            model.save(os.path.join(self.models_dir, 'audio_distress_model.h5'))
            self.model = model
            
            # Update distress labels to only include valid categories
            self.distress_labels = valid_categories
            
            print("Audio distress model trained successfully.")
            
        except Exception as e:
            print(f"Error training audio model: {str(e)}")
            self.model = None

    def start_audio_capture(self):
        """Start audio capture in a separate thread"""
        if self.is_capturing:
            return
        
        self.is_capturing = True
        self.audio_thread = threading.Thread(target=self._audio_capture_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        print("Audio capture started")

    def stop_audio_capture(self):
        """Stop audio capture"""
        self.is_capturing = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
            self.audio_thread = None
        print("Audio capture stopped")

    def _audio_capture_loop(self):
        """Audio capture loop running in a separate thread"""
        try:
            def audio_callback(indata, frames, time, status):
                """Callback for audio stream"""
                if status:
                    print(f"Audio status: {status}")
                
                # Convert to mono if stereo
                if indata.shape[1] > 1:
                    audio_data = indata[:, 0]
                else:
                    audio_data = indata[:, 0]
                
                # Add to buffer
                self.audio_buffer.extend(audio_data)
                
                # Process when buffer is full
                if len(self.audio_buffer) >= self.sample_rate * self.duration:
                    # Create a copy of the current buffer
                    audio_segment = np.array(list(self.audio_buffer))
                    
                    # Put in queue if not full
                    if not self.audio_queue.full():
                        self.audio_queue.put(audio_segment)
                    
                    # Clear half of the buffer to create overlap
                    for _ in range(int(self.sample_rate * self.duration / 2)):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
            
            # Start audio stream
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                              dtype='float32'):
                print(f"Audio stream started at {self.sample_rate}Hz")
                
                while self.is_capturing:
                    time.sleep(0.1)
                
        except Exception as e:
            print(f"Error in audio capture: {str(e)}")
            self.is_capturing = False

    def detect_distress(self):
        """Detect distress in captured audio"""
        if self.audio_queue.empty() or self.model is None:
            return None
        
        try:
            # Get audio data from queue
            audio_data = self.audio_queue.get()
            
            # Extract features
            features = self.extract_features(audio_data, self.sample_rate)
            features = np.expand_dims(features, axis=0)  # Add batch dimension
            
            # Get predictions
            predictions = self.model.predict(features, verbose=0)[0]
            
            # Convert to dictionary
            results = {}
            for i, label in enumerate(self.distress_labels):
                if i < len(predictions):
                    results[label] = float(predictions[i] * 100)
            
            # Generate audio visualization
            spectrogram = self.generate_spectrogram(audio_data)
            
            return {
                'distress_levels': results,
                'spectrogram': spectrogram,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Error detecting distress: {str(e)}")
            return None

    def generate_spectrogram(self, audio_data):
        """Generate spectrogram visualization of audio data"""
        try:
            plt.figure(figsize=(5, 2))
            plt.specgram(audio_data, Fs=self.sample_rate, NFFT=1024, noverlap=512)
            plt.axis('off')
            
            # Save to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
            
        except Exception as e:
            print(f"Error generating spectrogram: {str(e)}")
            return None

    def save_audio_sample(self, audio_data, label=None):
        """Save audio sample for further training"""
        try:
            # Create samples directory if it doesn't exist
            samples_dir = os.path.join(self.audio_dir, 'samples')
            os.makedirs(samples_dir, exist_ok=True)
            
            # Create label directory if provided
            if label and label in self.distress_labels:
                target_dir = os.path.join(samples_dir, label)
            else:
                target_dir = os.path.join(samples_dir, 'unlabeled')
            
            os.makedirs(target_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sample_{timestamp}.wav"
            filepath = os.path.join(target_dir, filename)
            
            # Save audio file
            wavfile.write(filepath, self.sample_rate, audio_data)
            print(f"Audio sample saved to {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"Error saving audio sample: {str(e)}")
            return None 