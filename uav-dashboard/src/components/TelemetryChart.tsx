import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface TelemetryPoint {
  timestamp: string;
  latitude: number;
  longitude: number;
  signal: number;
}

const TelemetryChart = () => {
  const [telemetryData, setTelemetryData] = useState<TelemetryPoint[]>([]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8000/api/telemetry/unity/latest");
        const data = await res.json();

        if (data?.timestamp) {
          setTelemetryData((prev) => [
            ...prev.slice(-29), // Keep last 30 points
            {
              timestamp: new Date(data.timestamp).toLocaleTimeString(),
              latitude: data.latitude,
              longitude: data.longitude,
              signal:
                data.signal === "low"
                  ? 1
                  : data.signal === "medium"
                  ? 2
                  : 3,
            },
          ]);
        }
      } catch (err) {
        console.error("Failed to fetch Unity telemetry:", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white p-4 rounded-lg shadow-lg">
      <h3 className="text-lg font-semibold mb-4">Signal Strength Over Time</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={telemetryData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis
            domain={[0, 3]}
            ticks={[1, 2, 3]}
            tickFormatter={(value) =>
              value === 1 ? "Low" : value === 2 ? "Medium" : "High"
            }
          />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="signal"
            stroke="#4f46e5"
            name="Signal Strength"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
    
  );
};

export default TelemetryChart;
