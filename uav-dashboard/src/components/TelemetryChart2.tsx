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
  battery_level: number;
  speed: number;
}

const TelemetryChart2 = () => {
  const [telemetryData, setTelemetryData] = useState<TelemetryPoint[]>([]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8000/api/telemetry/unity/latest");
        const data = await res.json();

        if (data?.timestamp) {
          setTelemetryData((prev) => [
            ...prev.slice(-29),
            {
              timestamp: new Date(data.timestamp).toLocaleTimeString(),
              battery_level: data.battery_level ?? 0,
              speed: data.speed ?? 0,
            },
          ]);
        }
      } catch (err) {
        console.error("⚠️ Failed to fetch telemetry:", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white p-4 rounded-lg shadow-lg col-span-1">
      <h3 className="text-lg font-semibold mb-4">Battery Drain vs Speed</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={telemetryData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" />
          <YAxis yAxisId="left" domain={[0, 100]} label={{ value: "Battery (%)", angle: -90, position: "insideLeft" }} />
          <YAxis yAxisId="right" orientation="right" domain={['auto', 'auto']} label={{ value: "Speed (m/s)", angle: 90, position: "insideRight" }} />
          <Tooltip />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="battery_level" stroke="#facc15" name="Battery Level" dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="speed" stroke="#3b82f6" name="Speed" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TelemetryChart2;
