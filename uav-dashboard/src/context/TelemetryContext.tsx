import React, { createContext, useContext, useState, useEffect } from 'react';

interface TelemetryData {
  altitude: number;
  speed: number;
  battery: number;
  latitude: number;
  longitude: number;
  heading: number;
  timestamp: number;
}

interface TelemetryContextType {
  telemetryData: TelemetryData;
  telemetryHistory: TelemetryData[];
  isConnected: boolean;
}

const TelemetryContext = createContext<TelemetryContextType | undefined>(undefined);

export const TelemetryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [telemetryData, setTelemetryData] = useState<TelemetryData>({
    altitude: 0,
    speed: 0,
    battery: 100,
    latitude: 51.505,
    longitude: -0.09,
    heading: 0,
    timestamp: Date.now()
  });
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryData[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Simulate real-time data updates
    const interval = setInterval(() => {
      const newData = {
        altitude: Math.random() * 100 + 100,
        speed: Math.random() * 30 + 10,
        battery: Math.max(0, telemetryData.battery - 0.1),
        latitude: telemetryData.latitude + (Math.random() - 0.5) * 0.001,
        longitude: telemetryData.longitude + (Math.random() - 0.5) * 0.001,
        heading: (telemetryData.heading + 5) % 360,
        timestamp: Date.now()
      };
      
      setTelemetryData(newData);
      setTelemetryHistory(prev => [...prev, newData].slice(-100));
      setIsConnected(true);
    }, 1000);

    return () => clearInterval(interval);
  }, [telemetryData]);

  return (
    <TelemetryContext.Provider value={{ telemetryData, telemetryHistory, isConnected }}>
      {children}
    </TelemetryContext.Provider>
  );
};

export const useTelemetry = () => {
  const context = useContext(TelemetryContext);
  if (context === undefined) {
    throw new Error('useTelemetry must be used within a TelemetryProvider');
  }
  return context;
};