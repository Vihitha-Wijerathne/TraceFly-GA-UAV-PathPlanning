from src.video.live_processor2 import LiveProcessor2

def main():
    try:
        # Initialize the biometric processor
        processor = LiveProcessor2()
        
        # Start processing webcam feed (use 0 for built-in webcam, 1 for external)
        print("Starting biometric detection...")
        print("Press 'q' to quit")
        processor.process_webcam(camera_id=0)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure your webcam is connected and accessible")
        print("2. Try a different camera_id if you have multiple cameras")
        print("3. Ensure good lighting conditions")
        print("4. Keep your face well-positioned in the frame")

if __name__ == "__main__":
    main() 