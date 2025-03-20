import cv2
import numpy as np
import librosa
from pathlib import Path
import tempfile
import subprocess
from datetime import datetime
import warnings
import soundfile as sf
from src.detection.thermal_detector import ThermalDetector
from src.detection.visual_detector import VisualDetector
from src.detection.audio_detector import AudioDetector

class VideoProcessor:
    def __init__(self):
        self.thermal_detector = ThermalDetector()
        self.visual_detector = VisualDetector()
        self.audio_detector = AudioDetector()
        
        # Load face detection and emotion recognition models
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Define emotions
        self.emotion_colors = {
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'neutral': (0, 255, 0),    # Green
            'surprise': (255, 0, 255),  # Purple
            'fear': (0, 165, 255)      # Orange
        }
        # Track emotions for each detected person
        self.person_emotions = {}
        self.emotion_memory = 10  # Number of frames to remember emotions
        
    def extract_audio(self, video_path):
        """Extract audio from video file using ffmpeg"""
        temp_audio = str(Path(tempfile.gettempdir()) / f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        
        try:
            # Extract audio using ffmpeg
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                temp_audio,
                '-y'
            ]
            
            subprocess.run(command, capture_output=True, check=True)
            return temp_audio
        except Exception as e:
            print(f"Failed to extract audio: {str(e)}")
            return None

    def process_video(self, video_path, output_path=None, display=False):
        """Process video file and detect objects/sounds"""
        if not output_path:
            output_path = str(Path(video_path).with_suffix('.processed.mp4'))

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract and load audio
        temp_audio = self.extract_audio(video_path)
        audio_frames = []
        audio_data = None
        sr = None

        if temp_audio:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio_data, sr = librosa.load(temp_audio, sr=None)
                    Path(temp_audio).unlink()  # Clean up temp file
            except Exception as e:
                print(f"Failed to load audio: {str(e)}")

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Process audio data
        if audio_data is not None:
            # Normalize audio
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            # Split into frames matching video fps
            samples_per_frame = int(sr / fps)
            audio_frames = [audio_data[i:i+samples_per_frame] 
                          for i in range(0, len(audio_data), samples_per_frame)]

        frame_number = 0
        prev_detections = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process visual detection
            visual_detections = self.visual_detector.detect_people(frame)
            
            # Process face emotions for each detection
            face_emotions = []
            for (x1, y1, x2, y2) in visual_detections:
                # Extract larger face region (upper 40% of body detection)
                face_height = int((y2 - y1) * 0.4)
                # Add margin for better face detection
                margin_x = int((x2 - x1) * 0.1)
                margin_y = int(face_height * 0.1)
                face_x1 = max(0, x1 - margin_x)
                face_y1 = max(0, y1 - margin_y)
                face_x2 = min(frame.shape[1], x2 + margin_x)
                face_y2 = min(frame.shape[0], y1 + face_height + margin_y)
                face_img = frame[face_y1:face_y2, face_x1:face_x2]
                
                if face_img.size > 0:
                    emotions = self.detect_face_emotion(face_img)
                    face_emotions.append(emotions if emotions else {})
                else:
                    face_emotions.append({})

            # Process current audio frame
            if frame_number < len(audio_frames):
                current_audio = audio_frames[frame_number]
                audio_features = self.audio_detector.extract_features_from_array(
                    current_audio, sr)
                current_emotions = self.audio_detector.detect_emotion(audio_features)
            else:
                current_emotions = {'scream': 0.0, 'fear': 0.0, 'distress': 0.0}

            # Update person emotions based on tracking
            self.update_person_emotions(visual_detections, prev_detections, current_emotions)
            prev_detections = visual_detections.copy()

            # Draw detections on frame
            frame = self.draw_detections(frame, visual_detections, face_emotions)

            # Write frame
            out.write(frame)

            if display:
                cv2.imshow('Processing Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_number += 1
            if frame_number % 30 == 0:
                print(f"Processing: {frame_number}/{total_frames} frames "
                      f"({frame_number/total_frames*100:.1f}%)")

        # Clean up
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        return output_path

    def update_person_emotions(self, current_detections, prev_detections, current_emotions):
        """Update emotion states for tracked persons"""
        # Clean up old tracks
        current_time = datetime.now()
        self.person_emotions = {k: v for k, v in self.person_emotions.items() 
                              if (current_time - v['last_seen']).seconds < self.emotion_memory}
        
        # Match current detections with previous ones
        for curr_box in current_detections:
            best_match = None
            min_dist = float('inf')
            curr_center = ((curr_box[0] + curr_box[2])/2, (curr_box[1] + curr_box[3])/2)
            
            # Find closest previous detection
            for prev_box in prev_detections:
                prev_center = ((prev_box[0] + prev_box[2])/2, (prev_box[1] + prev_box[3])/2)
                dist = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
                if dist < min_dist and dist < 50:  # Maximum distance threshold
                    min_dist = dist
                    best_match = prev_box
            
            # Update or create new emotion track
            box_id = f"{curr_box[0]}_{curr_box[1]}_{curr_box[2]}_{curr_box[3]}"
            if best_match is not None:
                prev_id = f"{best_match[0]}_{best_match[1]}_{best_match[2]}_{best_match[3]}"
                if prev_id in self.person_emotions:
                    # Update existing track
                    self.person_emotions[box_id] = self.person_emotions[prev_id]
                    self.person_emotions[box_id]['emotions'] = self.blend_emotions(
                        self.person_emotions[box_id]['emotions'], current_emotions)
                    self.person_emotions[box_id]['last_seen'] = current_time
                    if prev_id != box_id:
                        del self.person_emotions[prev_id]
            else:
                # Create new track
                self.person_emotions[box_id] = {
                    'emotions': current_emotions,
                    'last_seen': current_time
                }

    def blend_emotions(self, old_emotions, new_emotions, alpha=0.7):
        """Blend old and new emotions with smoothing"""
        return {
            k: alpha * old_emotions[k] + (1 - alpha) * new_emotions[k]
            for k in old_emotions
        }

    def detect_face_emotion(self, face_img):
        """Detect facial emotions using facial landmarks"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
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
                    
                    # Extract facial features
                    # Calculate histogram of oriented gradients (HOG)
                    face_roi = cv2.resize(face_roi, (64, 64))
                    gx = cv2.Sobel(face_roi, cv2.CV_32F, 1, 0)
                    gy = cv2.Sobel(face_roi, cv2.CV_32F, 0, 1)
                    mag, ang = cv2.cartToPolar(gx, gy)
                    
                    # Simple emotion detection based on facial features
                    # Horizontal gradient (smile/frown detection)
                    horizontal_gradient = np.mean(gx[32:48, 16:48])  # mouth region
                    # Vertical gradient (eyebrow position)
                    vertical_gradient = np.mean(gy[16:32, 16:48])    # eye region
                    # Overall intensity variation
                    intensity_var = np.var(face_roi)
                    
                    # Assign emotion probabilities based on features
                    if horizontal_gradient > 0:
                        emotions['happy'] = min(1.0, horizontal_gradient / 50)
                    elif horizontal_gradient < 0:
                        emotions['sad'] = min(1.0, -horizontal_gradient / 50)
                    
                    if vertical_gradient > 20:
                        emotions['surprise'] = min(1.0, vertical_gradient / 40)
                    elif vertical_gradient < -20:
                        emotions['angry'] = min(1.0, -vertical_gradient / 40)
                    
                    if intensity_var > 1000:
                        emotions['fear'] = min(1.0, intensity_var / 2000)
                    
                    # Neutral as fallback
                    if all(v < 0.2 for v in emotions.values()):
                        emotions['neutral'] = 0.7
                
                return emotions
            return None
        except Exception as e:
            print(f"Face emotion detection error: {str(e)}")
            return None

    def draw_detections(self, frame, visual_detections, face_emotions):
        """Draw detections and emotions on frame"""
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

        return frame