import { useEffect, useState } from "react";
import moment from "moment";

type LiDARHistoryRow = {
  timestamp: string;
  hit_count: number;
  tags: string[]; // New: all tags in that record
};

const classifyArea = (hits: number): string => {
  if (hits <= 3) return "Clear";
  if (hits <= 12) return "Few Obstacles";
  return "Heavy Obstacles";
};

const LiDARHistoryTable = () => {
  const [rows, setRows] = useState<LiDARHistoryRow[]>([]);

  useEffect(() => {
    const fetchData = () => {
      fetch("http://localhost:8000/api/lidar/unity/history/drone_test")
        .then((res) => res.json())
        .then((data: LiDARHistoryRow[]) => {
          const limited = data.slice(-100); // Keep only last 100 records
          setRows(limited);
        })
        .catch(console.error);
    };

    fetchData();
    const interval = setInterval(fetchData, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white p-4 rounded-lg shadow-md overflow-x-auto">
      <h3 className="text-lg font-semibold mb-4">LiDAR Data History</h3>
      <table className="min-w-full text-sm text-left border-collapse">
        <thead>
          <tr className="bg-gray-100 border-b">
            <th className="py-2 px-4 font-medium text-gray-700">Timestamp</th>
            <th className="py-2 px-4 font-medium text-gray-700">Hit Count</th>
            <th className="py-2 px-4 font-medium text-gray-700">
              Object Types
            </th>
            <th className="py-2 px-4 font-medium text-gray-700">Environment</th>
          </tr>
        </thead>
        <tbody>
          {rows
            .slice()
            .reverse()
            .map((row, i) => (
              <tr key={i} className="border-t hover:bg-gray-50">
                <td className="py-2 px-4 text-gray-800">
                  {moment(row.timestamp).format("YYYY-MM-DD HH:mm:ss")}
                </td>
                <td className="py-2 px-4 font-mono text-indigo-600">
                  {row.hit_count}
                </td>
                <td className="py-2 px-4">
                  {row.tags && row.tags.length > 0
                    ? Array.from(new Set(row.tags)).join(", ")
                    : "-"}
                </td>
                <td className="py-2 px-4">
                  <span
                    className={`inline-block px-2 py-1 rounded-full text-xs font-semibold ${
                      row.hit_count <= 3
                        ? "bg-green-100 text-green-700"
                        : row.hit_count <= 12
                        ? "bg-yellow-100 text-yellow-800"
                        : "bg-red-100 text-red-700"
                    }`}
                  >
                    {classifyArea(row.hit_count)}
                  </span>
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
};

export default LiDARHistoryTable;
