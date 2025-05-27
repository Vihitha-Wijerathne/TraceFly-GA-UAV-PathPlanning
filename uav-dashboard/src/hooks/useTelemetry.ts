import { useEffect, useState } from "react";

export interface TelemetryData {
  uav_id: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  altitude: number;
  speed: number;
  battery_level: number;
  signal: string;
}

export function useTelemetry() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const res = await fetch(
          "http://localhost:8000/api/telemetry/unity/latest"
        );
        const json = await res.json();
        setData(json);
      } catch (err) {
        console.error("Telemetry fetch error", err);
      }
    };

    const interval = setInterval(fetchTelemetry, 1000);
    return () => clearInterval(interval);
  }, []);

  return data;
}
