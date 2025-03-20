import React from 'react';
import { Link } from 'react-router-dom';
import { Bone as Drone, History, Settings } from 'lucide-react';
import { useTelemetry } from '../context/TelemetryContext';
import BiometricDetection from '../pages/BiometricDetection';

const Header = () => {
  const { isConnected } = useTelemetry();

  return (
    <header className="p-4 text-white bg-slate-800">
      <div className="container flex justify-between items-center mx-auto">
        <Link to="/" className="flex items-center space-x-2">
          <Drone size={24} />
          <span className="text-xl font-bold">UAV Telemetry</span>
        </Link>
        
        <nav className="flex items-center space-x-6">
          <Link to="/" className="flex items-center space-x-1 hover:text-blue-400">
            <Drone size={18} />
            <span>Dashboard</span>
          </Link>
          <Link to="/history" className="flex items-center space-x-1 hover:text-blue-400">
            <History size={18} />
            <span>History</span>
          </Link>
          <Link to="/settings" className="flex items-center space-x-1 hover:text-blue-400">
            <Settings size={18} />
            <span>Settings</span>
          </Link>
          <Link to="/biometrics" className="flex items-center space-x-1 hover:text-blue-400">
            <span>Biometrics</span>
          </Link>
          <Link to="/audio-distress" className="flex items-center space-x-1 hover:text-blue-400">
            <span>Audio Distress</span>
          </Link>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;