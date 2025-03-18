import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { format } from 'date-fns';
import { useTelemetry } from '../context/TelemetryContext';

const TelemetryChart = () => {
  const { telemetryHistory } = useTelemetry();

  const formatData = telemetryHistory.map(data => ({
    ...data,
    time: format(data.timestamp, 'HH:mm:ss')
  }));

  return (
    <div className="bg-white p-4 rounded-lg shadow-lg">
      <h3 className="text-lg font-semibold mb-4">Live Telemetry</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={formatData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="altitude"
            stroke="#8884d8"
            name="Altitude (m)"
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="speed"
            stroke="#82ca9d"
            name="Speed (m/s)"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="battery"
            stroke="#ffc658"
            name="Battery (%)"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TelemetryChart;