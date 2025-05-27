import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  Legend, CartesianGrid, ResponsiveContainer,
} from "recharts";
import moment from "moment";

const LiDARChart = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = () => {
      fetch("http://localhost:8000/api/lidar/unity/history/drone_test")
        .then(res => res.json())
        .then(allData => {
          const now = moment();
          const filtered = allData.filter((entry: any) =>
            moment(entry.timestamp).isAfter(now.clone().subtract(1, 'minute'))
          );
          setData(filtered);
        })
        .catch(console.error);
    };

    fetchData();
    const interval = setInterval(fetchData, 500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <h3 className="text-lg font-semibold mb-4">LiDAR Ray Hit Count</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" tickFormatter={t => moment(t).format("HH:mm:ss")} />
          <YAxis label={{ value: "Hits", angle: -90, position: "insideLeft" }} />
          <Tooltip labelFormatter={v => moment(v).format("HH:mm:ss")} />
          <Legend />
          <Line type="monotone" dataKey="hit_count" stroke="#6366f1" dot={{ r: 2 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LiDARChart;
