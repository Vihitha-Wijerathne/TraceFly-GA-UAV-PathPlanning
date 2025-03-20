import librosa
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import warnings
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from sklearn.preprocessing import StandardScaler

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1984, 64)  # 1984 = 64 * 31
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten while preserving batch size
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

class AudioDetector:
    def __init__(self):
        # More sensitive thresholds for voice detection
        self.energy_threshold = 0.05
        self.min_duration = 0.1
        
        # Rolling window parameters
        self.window_size = 5
        self.history = []
        
        self.emotion_thresholds = {
            'scream': {
                'spectral_centroid': 2000,  # Lower for better voice detection
                'spectral_rolloff': 3000,   # Lower for better voice detection
                'zero_crossing_rate': 0.2,  # Lower for better voice detection
                'min_duration': 0.2
            },
            'distress': {
                'rms_energy': 0.15,         # Lower for better voice detection
                'spectral_contrast': 0.3,   # Lower for better voice detection
                'min_duration': 0.15
            }
        }
        
        # Voice detection parameters
        self.voice_frequency_range = (85, 255)  # Hz, typical human voice range
        self.voice_energy_threshold = 0.03

    def _apply_rolling_average(self, emotions):
        """Apply rolling average to smooth emotion detection"""
        self.history.append(emotions)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Calculate average emotions
        avg_emotions = {}
        for emotion in emotions.keys():
            values = [h[emotion] for h in self.history]
            avg_emotions[emotion] = sum(values) / len(values)
        
        return avg_emotions

    def extract_features(self, audio_path):
        """Extract audio features including emotion-related features"""
        try:
            # Try multiple methods to load audio
            audio_data = None
            sample_rate = None
            
            # Method 1: Try soundfile (preferred method)
            try:
                audio_data, sample_rate = sf.read(audio_path)
            except Exception as e:
                print(f"soundfile failed: {str(e)}")

            # Method 2: Try scipy.io.wavfile
            if audio_data is None:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=WavFileWarning)
                        sample_rate, audio_data = wavfile.read(audio_path)
                        # Convert to float32 if needed
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                except Exception as e:
                    print(f"scipy.io.wavfile failed: {str(e)}")

            # Method 3: Try librosa as last resort
            if audio_data is None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        audio_data, sample_rate = librosa.load(audio_path)
                except Exception as e:
                    print(f"librosa failed: {str(e)}")

            if audio_data is None:
                raise RuntimeError("Failed to load audio file with any method")

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Normalize audio
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            
            # Extract emotion-related features
            features = {}
            
            # Spectral features
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate))
            features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=audio_data))
            
            # Temporal features
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
            features['spectral_contrast'] = np.mean(contrast)
            
            # Additional features for emotion detection
            features['mfccs'] = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13), axis=1)
            
            return audio_data, sample_rate, features

        except Exception as e:
            raise RuntimeError(f"Error processing audio file: {str(e)}. "
                             f"Supported formats: WAV, FLAC, OGG, MP3")

    def detect_emotion(self, features):
        """Detect emotions with improved reliability"""
        if features is None or not isinstance(features, dict):
            return {'scream': 0.0, 'distress': 0.0}
            
        emotions = {
            'scream': 0.0,
            'distress': 0.0
        }
        
        # Only process if voice is detected
        if not features.get('is_voice', False):
            return emotions
        
        # Check if audio is too short (safely get duration)
        duration = features.get('duration', 0)
        if duration < self.min_duration:
            return emotions
            
        # More lenient energy threshold for voice
        if features.get('rms_energy', 0) < self.voice_energy_threshold:
            return emotions
            
        # Detect scream with voice characteristics
        if (features.get('spectral_centroid', 0) > self.emotion_thresholds['scream']['spectral_centroid'] and
            features.get('spectral_rolloff', 0) > self.emotion_thresholds['scream']['spectral_rolloff'] and
            features.get('zero_crossing_rate', 0) > self.emotion_thresholds['scream']['zero_crossing_rate'] and
            duration >= self.emotion_thresholds['scream']['min_duration']):
            emotions['scream'] = min(1.0, features.get('voice_probability', 0) * 
                                       features.get('rms_energy', 0) * 2)
        
        # Detect distress with voice characteristics
        if (features.get('rms_energy', 0) > self.emotion_thresholds['distress']['rms_energy'] and
            features.get('spectral_contrast', 0) > self.emotion_thresholds['distress']['spectral_contrast'] and
            duration >= self.emotion_thresholds['distress']['min_duration']):
            emotions['distress'] = min(1.0, features.get('voice_probability', 0) * 
                                          features.get('rms_energy', 0) * 3)
        
        # Apply rolling average for smoothing
        smoothed_emotions = self._apply_rolling_average(emotions)
        
        return smoothed_emotions

    def predict(self, audio_path):
        """Detect audio distress and emotions"""
        audio_data, sr, features = self.extract_features(audio_path)
        
        # Calculate short-time energy
        frame_length = int(sr * 0.050)  # 50ms frames for better detection
        hop_length = int(sr * 0.025)    # 25ms hop
        
        # Calculate energy in frames
        frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
        energy = np.mean(frames ** 2, axis=0)
        
        # Also check for sudden changes in energy
        energy_diff = np.diff(energy)
        sudden_changes = np.abs(energy_diff) > self.energy_threshold
        
        # Detect high energy segments
        high_energy_frames = np.mean(energy > self.energy_threshold)
        
        # Print debug info
        print(f"Average energy: {np.mean(energy):.4f}")
        print(f"Max energy: {np.max(energy):.4f}")
        print(f"High energy frames: {high_energy_frames:.2%}")
        print(f"Sudden changes detected: {np.mean(sudden_changes):.2%}")
        
        # Detect emotions
        emotions = self.detect_emotion(features)
        print("\nEmotion Detection Results:")
        for emotion, score in emotions.items():
            print(f"{emotion.capitalize()}: {score:.2%}")
        
        return {
            'distress_detected': (high_energy_frames > 0.05) or (np.mean(sudden_changes) > 0.02),
            'emotions': emotions
        }

    def extract_features_from_array(self, audio_data, sample_rate):
        """Extract features from audio array with better preprocessing"""
        try:
            # Initialize features
            features = {
                'spectral_centroid': 0,
                'spectral_rolloff': 0,
                'spectral_bandwidth': 0,
                'spectral_flatness': 0,
                'zero_crossing_rate': 0,
                'rms_energy': 0,
                'spectral_contrast': 0,
                'duration': 0,
                'is_voice': False,
                'voice_probability': 0.0,
                'pitch': 0.0
            }
            
            if audio_data is None or len(audio_data) == 0:
                return features

            # Normalize audio
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            
            # Voice activity detection
            if len(audio_data) >= 512:  # Minimum length for pitch detection
                # Pitch detection
                pitches, magnitudes = librosa.piptrack(
                    y=audio_data, 
                    sr=sample_rate,
                    fmin=self.voice_frequency_range[0],
                    fmax=self.voice_frequency_range[1]
                )
                
                # Get the most prominent pitch
                pit_idx = magnitudes.argmax(axis=0)
                pitches_with_mag = pitches[pit_idx, range(pitches.shape[1])]
                features['pitch'] = np.mean(pitches_with_mag[pitches_with_mag > 0]) if len(pitches_with_mag) > 0 else 0
                
                # Check if pitch is in human voice range
                is_voice_pitch = (features['pitch'] >= self.voice_frequency_range[0] and 
                                features['pitch'] <= self.voice_frequency_range[1])
                
                # Voice probability based on multiple factors
                energy_factor = min(1.0, features['rms_energy'] / self.voice_energy_threshold)
                pitch_factor = 1.0 if is_voice_pitch else 0.0
                features['voice_probability'] = (energy_factor + pitch_factor) / 2
                features['is_voice'] = features['voice_probability'] > 0.5

            # Apply pre-emphasis to enhance high frequencies
            audio_data = librosa.effects.preemphasis(audio_data)
            
            # Apply noise reduction
            if len(audio_data) > sample_rate // 10:  # At least 100ms of audio
                noise_reduced = librosa.decompose.nn_filter(
                    audio_data.reshape(1, -1),
                    aggregate=np.median,
                    metric='cosine'
                ).reshape(-1)
                audio_data = noise_reduced
            
            # Enhanced spectral features
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate))
            features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(
                y=audio_data))
            
            # Enhanced temporal features
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(
                audio_data))
            features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
            
            # Enhanced spectral contrast with more bands
            contrast = librosa.feature.spectral_contrast(
                y=audio_data, 
                sr=sample_rate,
                n_bands=6,
                fmin=200.0
            )
            features['spectral_contrast'] = np.mean(contrast)
            
            # Add duration-based features
            features['duration'] = len(audio_data) / sample_rate
            
            return features
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            return features  # Return initialized features dictionary 