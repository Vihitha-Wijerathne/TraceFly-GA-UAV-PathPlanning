import React, { createContext, useContext, useEffect, useState } from "react";

export interface TelemetryData {
  uav_id: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  altitude: number;
  pitch: number;
  roll: number;
  yaw: number;
  speed: number;
  battery_level: number;
  wind: number;
  signal: string;
}

interface TelemetryContextType {
  telemetryData: TelemetryData | null;
  telemetryHistory: TelemetryData[];
  isConnected: boolean;
}

const TelemetryContext = createContext<TelemetryContextType | undefined>(
  undefined
);

export const TelemetryProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [telemetryData, setTelemetryData] = useState<TelemetryData | null>(
    null
  );
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryData[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(
          "http://localhost:8000/api/telemetry/unity/latest"
        );
        if (!res.ok) throw new Error("Failed to fetch telemetry");

        const data: TelemetryData = await res.json();
        setTelemetryData(data);
        setTelemetryHistory((prev) => [...prev, data].slice(-100));
        setIsConnected(true);
        console.log("[Telemetry] API data:", data);

      } catch (err) {
        console.error("[TelemetryContext] Error fetching data:", err);
        setIsConnected(false);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <TelemetryContext.Provider
      value={{ telemetryData, telemetryHistory, isConnected }}
    >
      {children}
    </TelemetryContext.Provider>
  );
};

export const useTelemetry = () => {
  const context = useContext(TelemetryContext);
  if (!context)
    throw new Error("useTelemetry must be used within TelemetryProvider");
  return context;
};
