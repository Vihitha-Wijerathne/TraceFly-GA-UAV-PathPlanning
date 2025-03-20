from flask import Flask, render_template, Response, jsonify, request
import threading
import queue
import time
import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import io
import base64
from collections import deque
import librosa
import csv
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Global variables
is_processing = False
processing_thread = None
audio_queue = queue.Queue(maxsize=10)
results_queue = queue.Queue(maxsize=10)
audio_buffer = deque(maxlen=48000)  # 3 seconds at 16kHz
last_results = None  # Store the last results for display after stopping

# Audio settings
SAMPLE_RATE = 16000
DURATION = 3  # seconds
DEVICE_ID = None  # Will be set by user selection

# Create directories
os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)
data_dir = os.path.join(app.root_path, 'data')
os.makedirs(data_dir, exist_ok=True)

# CSV file for storing distress levels
csv_file = os.path.join(data_dir, 'distress_levels.csv')

def audio_capture_loop():
    """Capture audio from microphone continuously"""
    global is_processing, last_results
    
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
            audio_buffer.extend(audio_data)
            
            # Process when buffer is full
            if len(audio_buffer) >= SAMPLE_RATE * DURATION:
                # Create a copy of the current buffer
                audio_segment = np.array(list(audio_buffer))
                
                # Put in queue if not full
                if not audio_queue.full():
                    audio_queue.put(audio_segment)
                
                # Clear half of the buffer to create overlap
                for _ in range(int(SAMPLE_RATE * DURATION / 2)):
                    if audio_buffer:
                        audio_buffer.popleft()
        
        # Start audio stream with selected device
        with sd.InputStream(callback=audio_callback,
                          channels=1,
                          samplerate=SAMPLE_RATE,
                          blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
                          dtype='float32',
                          device=DEVICE_ID):
            print(f"Audio stream started at {SAMPLE_RATE}Hz" + 
                 (f" on device {DEVICE_ID}" if DEVICE_ID is not None else ""))
            
            while is_processing:
                # Process audio data if available
                if not audio_queue.empty():
                    audio_data = audio_queue.get()
                    
                    # Generate spectrogram
                    spectrogram = generate_spectrogram(audio_data)
                    
                    # Analyze audio (simple volume detection)
                    volume_level = np.abs(audio_data).mean() * 100
                    
                    # Calculate frequency characteristics
                    freq_data = analyze_frequency(audio_data)
                    
                    # Create results
                    results = {
                        'spectrogram': spectrogram,
                        'volume_level': float(volume_level),
                        'timestamp': time.time(),
                        'distress_levels': {
                            'crying': freq_data['low_freq_energy'] * 100,
                            'screaming': freq_data['high_freq_energy'] * 100,
                            'shouting': freq_data['mid_freq_energy'] * 100,
                            'gasping': freq_data['rapid_changes'] * 100,
                            'choking': freq_data['irregular_pattern'] * 100,
                            'wheezing': freq_data['high_freq_variation'] * 100,
                            'normal': (1.0 - freq_data['overall_distress']) * 100
                        }
                    }
                    
                    # Save to CSV
                    save_to_csv(results)
                    
                    # Store last results
                    last_results = results
                    
                    # Put results in queue
                    if not results_queue.full():
                        results_queue.put(results)
                
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Error in audio capture: {str(e)}")
    finally:
        is_processing = False
        print("Audio capture stopped")

def analyze_frequency(audio_data):
    """Analyze frequency characteristics to estimate distress levels"""
    try:
        # Calculate FFT
        fft_data = np.abs(np.fft.rfft(audio_data))
        freqs = np.fft.rfftfreq(len(audio_data), 1/SAMPLE_RATE)
        
        # Normalize
        fft_data = fft_data / np.max(fft_data) if np.max(fft_data) > 0 else fft_data
        
        # Calculate energy in different frequency bands
        low_freq_energy = np.mean(fft_data[(freqs > 100) & (freqs < 500)])
        mid_freq_energy = np.mean(fft_data[(freqs > 500) & (freqs < 2000)])
        high_freq_energy = np.mean(fft_data[(freqs > 2000) & (freqs < 8000)])
        
        # Calculate variations
        high_freq_variation = np.std(fft_data[(freqs > 2000) & (freqs < 8000)])
        
        # Calculate rapid changes (zero crossings)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
        rapid_changes = zero_crossings
        
        # Calculate irregularity
        envelope = np.abs(librosa.stft(audio_data))
        envelope_std = np.std(np.mean(envelope, axis=0))
        irregular_pattern = envelope_std
        
        # Overall distress estimate
        overall_distress = (low_freq_energy + high_freq_energy + rapid_changes + irregular_pattern) / 4
        
        return {
            'low_freq_energy': low_freq_energy,
            'mid_freq_energy': mid_freq_energy,
            'high_freq_energy': high_freq_energy,
            'high_freq_variation': high_freq_variation,
            'rapid_changes': rapid_changes,
            'irregular_pattern': irregular_pattern,
            'overall_distress': overall_distress
        }
    except Exception as e:
        print(f"Error analyzing frequency: {str(e)}")
        return {
            'low_freq_energy': 0.1,
            'mid_freq_energy': 0.1,
            'high_freq_energy': 0.1,
            'high_freq_variation': 0.1,
            'rapid_changes': 0.1,
            'irregular_pattern': 0.1,
            'overall_distress': 0.1
        }

def save_to_csv(results):
    """Save distress levels to CSV file"""
    try:
        timestamp = datetime.fromtimestamp(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        distress_levels = results['distress_levels']
        
        # Create row data
        row = {
            'timestamp': timestamp,
            'volume_level': results['volume_level']
        }
        row.update(distress_levels)
        
        # Check if file exists
        file_exists = os.path.isfile(csv_file)
        
        # Write to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")

def generate_spectrogram(audio_data):
    """Generate spectrogram visualization of audio data"""
    try:
        plt.figure(figsize=(5, 2))
        plt.specgram(audio_data, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512)
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

def get_audio_devices():
    """Get list of available audio input devices"""
    devices = []
    try:
        device_list = sd.query_devices()
        for i, device in enumerate(device_list):
            if device['max_input_channels'] > 0:
                devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default': device.get('default_input', False)
                })
    except Exception as e:
        print(f"Error getting audio devices: {e}")
    
    return devices

@app.route('/')
def index():
    """Render main page"""
    return render_template('audio_distress.html')

@app.route('/get_audio_devices')
def get_devices():
    """Get available audio input devices"""
    devices = get_audio_devices()
    return jsonify({'devices': devices})

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """Start audio processing"""
    global is_processing, processing_thread, DEVICE_ID
    
    try:
        if not is_processing:
            # Get selected device ID if provided
            data = request.json or {}
            device_id = data.get('device_id', None)
            
            if device_id is not None:
                DEVICE_ID = int(device_id)
                print(f"Using audio device ID: {DEVICE_ID}")
            
            is_processing = True
            processing_thread = threading.Thread(target=audio_capture_loop)
            processing_thread.daemon = True
            processing_thread.start()
            return jsonify({'status': 'success', 'message': 'Audio processing started'})
        else:
            return jsonify({'status': 'warning', 'message': 'Audio processing already running'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error starting audio processing: {str(e)}'})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    """Stop audio processing"""
    global is_processing, processing_thread
    
    if is_processing:
        is_processing = False
        if processing_thread:
            processing_thread.join(timeout=1.0)
            processing_thread = None
        return jsonify({'status': 'success', 'message': 'Audio processing stopped'})
    else:
        return jsonify({'status': 'warning', 'message': 'Audio processing not running'})

@app.route('/get_results')
def get_results():
    """Get latest audio processing results"""
    global last_results
    
    if not results_queue.empty():
        last_results = results_queue.get()
        return jsonify(last_results)
    elif last_results:
        # Return last known results if queue is empty
        return jsonify(last_results)
    else:
        return jsonify({'status': 'no_data'})

@app.route('/get_history')
def get_history():
    """Get historical distress levels from CSV"""
    try:
        if os.path.exists(csv_file):
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Limit to last 100 entries
            if len(df) > 100:
                df = df.tail(100)
            
            # Convert to list of dictionaries
            history = df.to_dict('records')
            return jsonify({'status': 'success', 'history': history})
        else:
            return jsonify({'status': 'no_data', 'message': 'No history data available'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error getting history: {str(e)}'})

@app.route('/test_microphone', methods=['POST'])
def test_microphone():
    """Test microphone access and recording"""
    try:
        data = request.json or {}
        device_id = data.get('device_id', None)
        
        if device_id is not None:
            device_id = int(device_id)
        
        # Record a short audio clip
        duration = 3  # seconds
        
        print(f"Recording {duration} seconds of audio from device {device_id}...")
        recording = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            device=device_id,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        # Generate spectrogram
        spectrogram = generate_spectrogram(recording.flatten())
        
        # Analyze frequency
        freq_data = analyze_frequency(recording.flatten())
        
        # Create distress levels
        distress_levels = {
            'crying': freq_data['low_freq_energy'] * 100,
            'screaming': freq_data['high_freq_energy'] * 100,
            'shouting': freq_data['mid_freq_energy'] * 100,
            'gasping': freq_data['rapid_changes'] * 100,
            'choking': freq_data['irregular_pattern'] * 100,
            'wheezing': freq_data['high_freq_variation'] * 100,
            'normal': (1.0 - freq_data['overall_distress']) * 100
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Microphone test successful',
            'spectrogram': spectrogram,
            'distress_levels': distress_levels
        })
        
    except Exception as e:
        print(f"Error testing microphone: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error testing microphone: {str(e)}'
        })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear distress level history"""
    try:
        if os.path.exists(csv_file):
            os.remove(csv_file)
        return jsonify({'status': 'success', 'message': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error clearing history: {str(e)}'})

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5001) 