from flask import Flask, render_template, Response
import cv2
from src.video.live_processor2 import LiveProcessor2
import threading
import queue
import time
import numpy as np
from scipy import fftpack, ndimage
import os
import h5py

app = Flask(__name__)

# Increase queue size for better buffering
frame_queue = queue.Queue(maxsize=10)
processor = LiveProcessor2()

# Global variables for landmark smoothing
last_landmarks = None
landmark_smoothing_factor = 0.3  # Reduced for faster updates
last_face_landmarks = None
face_smoothing_factor = 0.2  # Reduced for faster updates

def smooth_landmarks(current, last, smoothing_factor):
    """Smooth landmarks to reduce jitter"""
    if last is None:
        return current
    if current is None:
        return last
    
    smoothed = []
    for i in range(len(current)):
        smooth_point = np.array([
            current[i].x * (1 - smoothing_factor) + last[i].x * smoothing_factor,
            current[i].y * (1 - smoothing_factor) + last[i].y * smoothing_factor,
            current[i].z * (1 - smoothing_factor) + last[i].z * smoothing_factor
        ])
        current[i].x, current[i].y, current[i].z = smooth_point
    return current

def process_camera():
    """Process camera feed with optimized performance"""
    global last_landmarks, last_face_landmarks
    
    # Initialize camera with optimized settings for higher FPS
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increased buffer
    cap.set(cv2.CAP_PROP_FPS, 60)  # Increased FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    frame_count = 0
    last_emotions = None
    last_pose = None
    last_injuries = []
    
    # Performance tracking
    start_time = time.time()
    fps_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every 30 frames
            if fps_counter >= 30:
                elapsed = time.time() - start_time
                fps = fps_counter / elapsed
                print(f"Processing FPS: {fps:.1f}")
                fps_counter = 0
                start_time = time.time()
            
            # Process people detection every frame
            visual_detections = processor.visual_detector.detect_people(frame)
            
            # Convert to RGB once for all processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect injuries every 10 frames (reduced frequency)
            if frame_count % 10 == 0:
                injuries = processor.detect_injuries(frame)
                if injuries:
                    last_injuries = injuries
            
            # Draw injuries from last detection
            if last_injuries:
                processor.draw_injuries(frame, last_injuries)
            
            # Process people in the frame
            for i, (x1, y1, x2, y2) in enumerate(visual_detections):
                if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                    continue

                # Draw person rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get face region
                face_roi = frame[y1:y2, x1:x2]
                
                # Process face landmarks every 3rd frame
                if frame_count % 3 == 0 and face_roi.size > 0:
                    face_mesh_results = processor.face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    if face_mesh_results.multi_face_landmarks:
                        # Smooth face landmarks
                        current_face_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
                        if last_face_landmarks is not None:
                            current_face_landmarks = smooth_landmarks(
                                current_face_landmarks, 
                                last_face_landmarks, 
                                face_smoothing_factor
                            )
                        last_face_landmarks = current_face_landmarks
                        
                        # Draw facial emotions
                        processor.draw_facial_emotions(frame, face_roi, x1, y1, x2, y2)
                
                # Process pose every 5th frame (reduced frequency)
                if frame_count % 5 == 0:
                    pose_results = processor.pose.process(frame_rgb)
                    if pose_results.pose_landmarks:
                        # Smooth pose landmarks
                        current_landmarks = pose_results.pose_landmarks.landmark
                        if last_landmarks is not None:
                            current_landmarks = smooth_landmarks(
                                current_landmarks, 
                                last_landmarks, 
                                landmark_smoothing_factor
                            )
                        last_landmarks = current_landmarks
                        
                        # Update pose
                        last_pose = processor.classify_pose(pose_results.pose_landmarks)
                        
                        # Draw pose skeleton
                        processor.mp_drawing.draw_landmarks(
                            frame,
                            pose_results.pose_landmarks,
                            processor.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=processor.mp_drawing.DrawingSpec(
                                color=(255, 0, 0), thickness=2, circle_radius=2),
                            connection_drawing_spec=processor.mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=1)
                        )
                
                # Process emotions every 4th frame (reduced frequency)
                if frame_count % 4 == 0 and face_roi.size > 0:
                    emotions = processor.detect_emotions(face_roi)
                    if emotions:
                        last_emotions = emotions
                
                # Draw labels
                label_parts = [f"Person {i+1}"]
                
                if last_emotions:
                    top_emotions = dict(sorted(last_emotions.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:2])
                    for emotion, score in top_emotions.items():
                        if score > 20:
                            label_parts.append(f"{emotion.capitalize()} ({score:.0f}%)")
                
                if last_pose:
                    label_parts.append(last_pose.replace('_', ' ').title())
                
                # Draw labels efficiently
                y_offset = y1 - 10
                for j, part in enumerate(label_parts):
                    color = (0, 255, 0)
                    if j > 0 and j < len(label_parts) - 1:
                        emotion = part.split()[0].lower()
                        color = processor.get_emotion_color(emotion)
                    elif j == len(label_parts) - 1 and last_pose:
                        color = processor.pose_colors.get(last_pose.split('_')[0], (0, 255, 0))
                    
                    cv2.putText(frame, part, (x1, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_offset -= 25

            # Put processed frame in queue without blocking
            if not frame_queue.full():
                frame_queue.put(frame)
            
            # No sleep to maximize frame rate

    finally:
        cap.release()

def generate_frames():
    """Generate frames for streaming with higher quality"""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Higher quality JPEG encoding
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Shorter wait time
            time.sleep(0.005)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start camera processing in a separate thread with higher priority
    camera_thread = threading.Thread(target=process_camera, daemon=True)
    camera_thread.start()
    
    # Run Flask app with optimized settings
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)