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

const TelemetryChart = () => {
  const [telemetryData, setTelemetryData] = useState([]);

  useEffect(() => {
    // Fetch telemetry data from the backend
    fetch("http://localhost:8000/api/telemetry/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latitude: [6.9271, 6.9272, 6.9273],
        longitude: [79.8612, 79.8613, 79.8614],
        imu: [
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9],
        ],
        signal_strength: [80, 70, 90],
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        const chartData = data.prioritized_data.latitude.map(
          (lat: number, index: number) => ({
            latitude: lat,
            longitude: data.prioritized_data.longitude[index],
            signal_strength: data.prioritized_data.signal_strength[index],
          })
        );
        setTelemetryData(chartData);
      })
      .catch((error) => console.error("Error fetching telemetry data:", error));
  }, []);

  return (
    <div className="bg-white p-4 rounded-lg shadow-lg">
      <h3 className="text-lg font-semibold mb-4">Processed Telemetry</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={telemetryData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="latitude" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="signal_strength"
            stroke="#8884d8"
            name="Signal Strength"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TelemetryChart;
