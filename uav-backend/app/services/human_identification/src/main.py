from src.fusion.detection_fusion import DetectionFusion
from src.visualization.visualizer import DetectionVisualizer
from src.video.video_processor import VideoProcessor
from src.video.live_processor import LiveProcessor
import cv2
import urllib.request
import os
import ssl
import certifi
import kaggle
from pathlib import Path
import shutil

def main():
    # Initialize fusion system
    fusion_system = DetectionFusion()
    
    try:
        # Process sample scene
        results = fusion_system.process_scene(
            'data/thermal/Sample.png',
            'data/visual/Sample.jpg',
            'data/audio/Sample.wav'
        )
        
        # Load visual image for visualization
        visual_img = cv2.imread('data/visual/sample.jpg')
        
        # Visualize results
        visualizer = DetectionVisualizer()
        fig = visualizer.visualize_detections(
            results['thermal_img'],
            visual_img,
            results
        )
        
        # Save or display the visualization
        fig.savefig('output/detection_results.png')
        
    except Exception as e:
        print(f"Error running detection system: {str(e)}")

def download_sample_data():
    """Download sample data using Kaggle API"""
    try:
        # Create directories if they don't exist
        os.makedirs('data/thermal', exist_ok=True)
        os.makedirs('data/visual', exist_ok=True)
        os.makedirs('data/audio', exist_ok=True)
        os.makedirs('data/video', exist_ok=True)
        os.makedirs('output', exist_ok=True)

        print("Authenticating with Kaggle...")
        kaggle.api.authenticate()

        # Download video dataset
        print("Downloading video dataset...")
        kaggle.api.dataset_download_files(
            'uwrfkaggler/ravdess-emotional-speech-audio',
            path='data/download/video',
            unzip=True
        )

        # 1. Download FLIR Thermal Dataset
        print("Downloading thermal dataset...")
        kaggle.api.dataset_download_files(
            'deepnewbie/flir-thermal-images-dataset',
            path='data/download/thermal',
            unzip=True
        )

        # 2. Download Visual Dataset (COCO person detection)
        print("Downloading visual dataset...")
        kaggle.api.dataset_download_files(
            'valentynsichkar/person-detection-dataset',
            path='data/download/visual',
            unzip=True
        )

        # 3. Download Audio Dataset (Urban Sound)
        print("Downloading audio dataset...")
        kaggle.api.dataset_download_files(
            'chrisfilo/urbansound8k',
            path='data/download/audio',
            unzip=True
        )

        # Process and copy sample files
        # Thermal sample
        thermal_dir = Path('data/download/thermal/thermal_8_bit')
        if thermal_dir.exists():
            thermal_files = list(thermal_dir.glob('*.jpg'))
            if thermal_files:
                shutil.copy(thermal_files[0], 'data/thermal/sample.jpg')
                print(f"Copied thermal sample: {thermal_files[0].name}")

        # Visual sample
        visual_dir = Path('data/download/visual/images')
        if visual_dir.exists():
            visual_files = list(visual_dir.glob('*.jpg'))
            if visual_files:
                shutil.copy(visual_files[0], 'data/visual/sample.jpg')
                print(f"Copied visual sample: {visual_files[0].name}")

        # Audio sample
        audio_dir = Path('data/download/audio/fold1')
        if audio_dir.exists():
            audio_files = list(audio_dir.glob('*.wav'))
            if audio_files:
                shutil.copy(audio_files[0], 'data/audio/sample.wav')
                print(f"Copied audio sample: {audio_files[0].name}")

        # Copy sample video
        video_dir = Path('data/download/video')
        if video_dir.exists():
            video_files = list(video_dir.glob('*.mp4'))
            if video_files:
                shutil.copy(video_files[0], 'data/video/sample.mp4')
                print(f"Copied video sample: {video_files[0].name}")

        # Clean up downloaded files
        print("Cleaning up downloaded files...")
        shutil.rmtree('data/download', ignore_errors=True)

        print("Sample data download complete")

    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nPlease ensure you have configured your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")

def process_video(video_path, output_path=None):
    """Process a video file with all detection modalities"""
    processor = VideoProcessor()
    try:
        output_path = processor.process_video(
            video_path,
            output_path=output_path,
            display=True  # Set to False to disable live preview
        )
        print(f"Processed video saved to: {output_path}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

def process_webcam(camera_id=0):
    """Process live webcam feed with emotion detection"""
    processor = LiveProcessor()
    try:
        processor.process_webcam(camera_id)
    except Exception as e:
        print(f"Error processing webcam: {str(e)}")

if __name__ == "__main__":
    # download_sample_data()
    main()
    # Uncomment to process video:
    process_video('data/video/sample.mp4')
    # Uncomment to start webcam processing:
   # process_webcam()  # Use process_webcam(1) for external webcam 