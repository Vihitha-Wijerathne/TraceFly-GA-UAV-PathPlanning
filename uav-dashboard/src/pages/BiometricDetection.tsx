 import React, { useState, useEffect } from 'react';

const BiometricDetection: React.FC = () => {
  const [cameraStatus, setCameraStatus] = useState<string>('Connecting...');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    const checkServerConnection = async () => {
      try {
        // Try to connect to the Flask server
        const response = await fetch('http://localhost:5000/', { 
          method: 'HEAD',
          mode: 'no-cors' // This is needed since the Flask server might not have CORS enabled
        });
        
        setCameraStatus('Active');
        setIsLoading(false);
      } catch (err) {
        setError('Could not connect to biometric server. Please ensure it is running.');
        setCameraStatus('Offline');
        setIsLoading(false);
      }
    };

    checkServerConnection();
  }, []);

  return (
    <div className="container mx-auto px-4">
      <h1 className="text-2xl font-bold mb-6 text-center">Biometric Detection System</h1>
      
      <div className="bg-white rounded-lg shadow-md p-6 max-w-3xl mx-auto">
        {error ? (
          <div className="p-4 bg-red-100 text-red-700 rounded-md mb-4">
            {error}
          </div>
        ) : null}
        
        <div className="relative w-full">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-75 rounded-md">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
          )}
          
          {/* Video feed from Flask backend */}
          <img 
            src="http://localhost:5000/video_feed" 
            className="w-full h-auto rounded-md"
            alt="Video feed"
            onLoad={() => {
              setCameraStatus('Active');
              setIsLoading(false);
            }}
            onError={() => {
              setError('Failed to load video feed. Please check the server connection.');
              setCameraStatus('Error');
              setIsLoading(false);
            }}
          />
        </div>
        
        <div className={`mt-4 p-3 rounded-md text-center ${
          cameraStatus === 'Active' ? 'bg-blue-50' : 
          cameraStatus === 'Error' ? 'bg-red-50' : 'bg-yellow-50'
        }`}>
          Camera Status: <span className="font-semibold">{cameraStatus}</span>
        </div>
      </div>
    </div>
  );
};

export default BiometricDetection;