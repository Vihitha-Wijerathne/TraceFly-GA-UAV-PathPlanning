import React, { useState, useEffect, useRef } from 'react';
import { AlertTriangle, Play, Square, RefreshCw, CheckCircle, Brain, Mic, Activity } from 'lucide-react';
import { History } from 'lucide-react';

interface DistressLevel {
  [key: string]: number;
}

interface AlertItem {
  type: string;
  value: number;
  timestamp: number;
}

const AudioDistress: React.FC = () => {
  // State variables
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [cameraStatus, setCameraStatus] = useState<string>('Inactive');
  const [spectrogramSrc, setSpectrogramSrc] = useState<string>('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=');
  const [distressLevels, setDistressLevels] = useState<DistressLevel>({});
  const [currentAlert, setCurrentAlert] = useState<AlertItem | null>(null);
  const [alertHistory, setAlertHistory] = useState<AlertItem[]>([]);
  const [alertCount, setAlertCount] = useState<number>(0);
  const [notification, setNotification] = useState<{message: string, type: string} | null>(null);
  const [audioDevices, setAudioDevices] = useState<{id: string, name: string, default?: boolean}[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [isTesting, setIsTesting] = useState<boolean>(false);
  
  const pollingIntervalRef = useRef<number | null>(null);
  const alertThreshold = 70;

  // Load audio devices
  const loadAudioDevices = async () => {
    try {
      const response = await fetch('http://localhost:5001/get_audio_devices');
      const data = await response.json();
      
      if (data.devices && data.devices.length > 0) {
        setAudioDevices(data.devices);
        showNotification(`Found ${data.devices.length} microphone devices`, 'info');
      } else {
        showNotification('No microphone devices found', 'warning');
      }
    } catch (error) {
      console.error('Error loading microphone devices:', error);
      showNotification('Error loading microphone devices', 'error');
    }
  };

  // Test microphone
  const testMicrophone = async () => {
    setIsTesting(true);
    try {
      const response = await fetch('http://localhost:5001/test_microphone', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ device_id: selectedDevice })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        showNotification(data.message);
        
        if (data.spectrogram) {
          setSpectrogramSrc(`data:image/png;base64,${data.spectrogram}`);
        }
      } else {
        showNotification(data.message, 'warning');
      }
    } catch (error) {
      console.error('Error testing microphone:', error);
      showNotification('Error testing microphone', 'error');
    } finally {
      setIsTesting(false);
    }
  };

  // Start audio processing
  const startProcessing = async () => {
    try {
      const response = await fetch('http://localhost:5001/start_processing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ device_id: selectedDevice })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setIsProcessing(true);
        setCameraStatus('Active');
        showNotification('Audio processing started');
        
        // Start polling for results - use window.setInterval instead
        pollingIntervalRef.current = window.setInterval(fetchResults, 500);
      } else {
        showNotification(data.message, 'warning');
      }
    } catch (error) {
      console.error('Error starting processing:', error);
      showNotification('Error starting audio processing', 'error');
    }
  };

  // Stop audio processing
  const stopProcessing = async () => {
    try {
      const response = await fetch('http://localhost:5001/stop_processing', { 
        method: 'POST' 
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setIsProcessing(false);
        setCameraStatus('Inactive');
        showNotification('Audio processing stopped');
        
        // Stop polling - use window.clearInterval
        if (pollingIntervalRef.current !== null) {
          window.clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      } else {
        showNotification(data.message, 'warning');
      }
    } catch (error) {
      console.error('Error stopping processing:', error);
      showNotification('Error stopping audio processing', 'error');
    }
  };

  // Fetch results from server
  const fetchResults = async () => {
    try {
      const response = await fetch('http://localhost:5001/get_results');
      const data = await response.json();
      
      if (data.status === 'no_data') {
        return;
      }
      
      // Update spectrogram
      if (data.spectrogram) {
        setSpectrogramSrc(`data:image/png;base64,${data.spectrogram}`);
      }
      
      // Update distress levels
      if (data.distress_levels) {
        setDistressLevels(data.distress_levels);
        
        // Check for alerts
        checkForAlerts(data.distress_levels, data.timestamp);
      }
    } catch (error) {
      console.error('Error fetching results:', error);
    }
  };

  // Check for alerts
  const checkForAlerts = (levels: DistressLevel, timestamp: number) => {
    // Find highest distress level
    const entries = Object.entries(levels)
      .filter(([type, _]) => type !== 'normal') // Exclude 'normal' category
      .sort((a, b) => b[1] - a[1]);
    
    if (entries.length > 0) {
      const [type, value] = entries[0];
      
      if (value >= alertThreshold) {
        const newAlert = { type, value, timestamp };
        
        // Update current alert
        setCurrentAlert(newAlert);
        
        // Add to history
        setAlertHistory(prev => [newAlert, ...prev.slice(0, 9)]);
        
        // Update alert count
        setAlertCount(prev => prev + 1);
        
        // Play alert sound
        try {
          const audio = new Audio('/static/alert.mp3');
          audio.play();
        } catch (e) {
          console.log('Error playing alert sound:', e);
        }
      }
    }
  };

  // Save audio sample
  const saveAudioSample = async (label: string) => {
    try {
      const response = await fetch('http://localhost:5001/save_sample', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ label })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        showNotification(data.message);
      } else {
        showNotification(data.message, 'warning');
      }
    } catch (error) {
      console.error('Error saving sample:', error);
      showNotification('Error saving audio sample', 'error');
    }
  };

  // Retrain model
  const retrainModel = async () => {
    if (window.confirm('Are you sure you want to retrain the model? This may take some time.')) {
      setIsTraining(true);
      
      try {
        const response = await fetch('http://localhost:5000/retrain_model', { 
          method: 'POST' 
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
          showNotification(data.message);
        } else {
          showNotification(data.message, 'error');
        }
      } catch (error) {
        console.error('Error retraining model:', error);
        showNotification('Error retraining model', 'error');
      } finally {
        setIsTraining(false);
      }
    }
  };

  // Show notification
  const showNotification = (message: string, type: string = 'success') => {
    setNotification({ message, type });
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
      setNotification(null);
    }, 3000);
  };

  // Format timestamp
  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
  };

  // Get color class for distress levels
  const getColorClass = (value: number) => {
    if (value >= 80) return 'bg-red-500';
    if (value >= 60) return 'bg-orange-500';
    if (value >= 40) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  // Load audio devices on component mount
  useEffect(() => {
    loadAudioDevices();
    
    // Cleanup on unmount
    return () => {
      if (pollingIntervalRef.current !== null) {
        window.clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  return (
    <div className="container px-4 mx-auto">
      {/* Notification */}
      {notification && (
        <div className={`fixed top-5 right-5 z-50 p-4 rounded-md shadow-md ${
          notification.type === 'success' ? 'bg-green-100 text-green-800' :
          notification.type === 'warning' ? 'bg-yellow-100 text-yellow-800' :
          notification.type === 'info' ? 'bg-blue-100 text-blue-800' :
          'bg-red-100 text-red-800'
        }`}>
          {notification.message}
        </div>
      )}
      
      <h1 className="mb-6 text-2xl font-bold text-center">Audio Distress Detection System</h1>
      
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Left Column (spans 2 columns on large screens) */}
        <div className="space-y-6 lg:col-span-2">
          {/* Microphone Setup Card */}
          <div className="overflow-hidden bg-white rounded-lg shadow-md">
            <div className="flex items-center p-4 font-semibold text-white bg-gray-800">
              <Mic className="mr-2" size={18} />
              Microphone Setup
            </div>
            <div className="p-4">
              <div className="p-4 mb-4 bg-gray-100 rounded-md">
                <div className="mb-3">
                  <label htmlFor="mic-select" className="block mb-2 font-medium">
                    Select Microphone:
                  </label>
                  <select 
                    id="mic-select" 
                    className="p-2 w-full rounded-md border"
                    value={selectedDevice}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    disabled={isProcessing}
                  >
                    <option value="">Default Microphone</option>
                    {audioDevices.map(device => (
                      <option key={device.id} value={device.id}>
                        {device.name}{device.default ? ' (Default)' : ''}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button 
                    className="flex items-center px-4 py-2 text-white bg-gray-500 rounded-md"
                    onClick={loadAudioDevices}
                    disabled={isProcessing}
                  >
                    <RefreshCw className="mr-2" size={16} />
                    Refresh Devices
                  </button>
                  <button 
                    className="flex items-center px-4 py-2 text-white bg-blue-500 rounded-md"
                    onClick={testMicrophone}
                    disabled={isProcessing || isTesting}
                  >
                    {isTesting ? (
                      <>
                        <div className="mr-2 w-4 h-4 rounded-full border-2 border-white animate-spin border-t-transparent"></div>
                        Testing...
                      </>
                    ) : (
                      <>
                        <CheckCircle className="mr-2" size={16} />
                        Test Microphone
                      </>
                    )}
                  </button>
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <div className={`w-3 h-3 rounded-full mr-2 ${
                    isProcessing ? 'bg-green-500 shadow-lg shadow-green-200' : 'bg-red-500'
                  }`}></div>
                  <span>{cameraStatus}</span>
                </div>
                <div className="flex gap-2">
                  <button 
                    className="flex items-center px-4 py-2 text-white bg-green-500 rounded-md"
                    onClick={startProcessing}
                    disabled={isProcessing}
                  >
                    <Play className="mr-2" size={16} />
                    Start
                  </button>
                  <button 
                    className="flex items-center px-4 py-2 text-white bg-red-500 rounded-md"
                    onClick={stopProcessing}
                    disabled={!isProcessing}
                  >
                    <Square className="mr-2" size={16} />
                    Stop
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          {/* Spectrogram Card */}
          <div className="overflow-hidden bg-white rounded-lg shadow-md">
            <div className="flex items-center p-4 font-semibold text-white bg-gray-800">
              <Activity className="mr-2" size={18} />
              Audio Spectrogram
            </div>
            <div className="p-4">
              <div className="p-2 text-center bg-gray-800 rounded-md">
                <img 
                  src={spectrogramSrc} 
                  className="max-w-full h-auto rounded-md"
                  alt="Audio Spectrogram" 
                />
              </div>
            </div>
          </div>
          
          {/* Distress Levels Card */}
          <div className="overflow-hidden bg-white rounded-lg shadow-md">
            <div className="flex items-center p-4 font-semibold text-white bg-gray-800">
              <AlertTriangle className="mr-2" size={18} />
              Distress Levels
            </div>
            <div className="p-4">
              {Object.entries(distressLevels)
                .sort((a, b) => b[1] - a[1])
                .map(([type, value]) => (
                  <div key={type} className="mb-3">
                    <div className="flex justify-between mb-1">
                      <span>{type.charAt(0).toUpperCase() + type.slice(1)}</span>
                      <span>{Math.round(value)}%</span>
                    </div>
                    <div className="w-full h-8 bg-gray-200 rounded-full">
                      <div 
                        className={`h-full rounded-full ${getColorClass(value)}`}
                        style={{ width: `${value}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              
              {Object.keys(distressLevels).length === 0 && (
                <div className="py-4 text-center text-gray-500">
                  No distress data available
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Right Column */}
        <div className="space-y-6">
          {/* Alert Card */}
          <div className="overflow-hidden relative bg-white rounded-lg shadow-md">
            {alertCount > 0 && (
              <div className="flex absolute -top-2 -right-2 justify-center items-center w-6 h-6 text-xs font-bold text-white bg-red-500 rounded-full">
                {alertCount}
              </div>
            )}
            <div className="flex items-center p-4 font-semibold text-white bg-gray-800">
              <AlertTriangle className="mr-2" size={18} />
              Current Alert
            </div>
            <div className="p-4">
              {currentAlert ? (
                <div className="p-4 text-red-800 bg-red-100 rounded-md">
                  <div className="flex justify-between items-center">
                    <div>
                      <strong>{currentAlert.type.toUpperCase()} DETECTED!</strong>
                      <div>Confidence: {Math.round(currentAlert.value)}%</div>
                    </div>
                    <AlertTriangle size={32} className="text-red-500" />
                  </div>
                </div>
              ) : (
                <div className="p-4 text-blue-800 bg-blue-100 rounded-md">
                  No active alerts
                </div>
              )}
            </div>
          </div>
          
          {/* History Card */}
          <div className="overflow-hidden bg-white rounded-lg shadow-md">
            <div className="flex items-center p-4 font-semibold text-white bg-gray-800">
              <History className="mr-2" size={18} />
              Alert History
            </div>
            <div className="p-4">
              {alertHistory.length > 0 ? (
                <div className="space-y-2">
                  {alertHistory.map((alert, index) => (
                    <div key={index} className="py-2 pl-3 bg-gray-50 rounded-r-md border-l-4 border-red-500">
                      <div><strong>{alert.type.toUpperCase()}</strong> ({Math.round(alert.value)}%)</div>
                      <div className="text-xs text-gray-500">{formatTimestamp(alert.timestamp)}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-4 text-center text-gray-500">
                  No alert history yet
                </div>
              )}
            </div>
          </div>
          
          {/* Training Card */}
          <div className="overflow-hidden bg-white rounded-lg shadow-md">
            <div className="flex items-center p-4 font-semibold text-white bg-gray-800">
              <Brain className="mr-2" size={18} />
              Model Training
            </div>
            <div className="p-4">
              <p className="mb-2">Save current audio sample as:</p>
              <div className="flex flex-wrap gap-2 mb-4">
                <button 
                  className="px-2 py-1 text-sm text-red-500 rounded-md border border-red-500 hover:bg-red-50"
                  onClick={() => saveAudioSample('crying')}
                >
                  Crying
                </button>
                <button 
                  className="px-2 py-1 text-sm text-red-500 rounded-md border border-red-500 hover:bg-red-50"
                  onClick={() => saveAudioSample('screaming')}
                >
                  Screaming
                </button>
                <button 
                  className="px-2 py-1 text-sm text-red-500 rounded-md border border-red-500 hover:bg-red-50"
                  onClick={() => saveAudioSample('shouting')}
                >
                  Shouting
                </button>
                <button 
                  className="px-2 py-1 text-sm text-yellow-500 rounded-md border border-yellow-500 hover:bg-yellow-50"
                  onClick={() => saveAudioSample('gasping')}
                >
                  Gasping
                </button>
                <button 
                  className="px-2 py-1 text-sm text-yellow-500 rounded-md border border-yellow-500 hover:bg-yellow-50"
                  onClick={() => saveAudioSample('choking')}
                >
                  Choking
                </button>
                <button 
                  className="px-2 py-1 text-sm text-yellow-500 rounded-md border border-yellow-500 hover:bg-yellow-50"
                  onClick={() => saveAudioSample('wheezing')}
                >
                  Wheezing
                </button>
                <button 
                  className="px-2 py-1 text-sm text-green-500 rounded-md border border-green-500 hover:bg-green-50"
                  onClick={() => saveAudioSample('normal')}
                >
                  Normal
                </button>
              </div>
              <button 
                className="flex justify-center items-center py-2 w-full text-white bg-blue-500 rounded-md"
                onClick={retrainModel}
                disabled={isTraining}
              >
                {isTraining ? (
                  <>
                    <div className="mr-2 w-4 h-4 rounded-full border-2 border-white animate-spin border-t-transparent"></div>
                    Retraining...
                  </>
                ) : (
                  <>
                    <RefreshCw className="mr-2" size={16} />
                    Retrain Model
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AudioDistress; 