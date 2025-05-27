import LiDARChart from "../components/LiDARChart";
import LiDARHistoryTable from "../components/LiDARHistoryTable";

const LiDARData = () => {
  return (
    <div className="container mx-auto p-6 space-y-10">
      <h2 className="text-2xl font-bold text-gray-800 flex justify-center">📡 LiDAR Analytics</h2>

      {/* Center the chart */}
      <div className="flex justify-center">
        <div className="w-full max-w-4xl">
          <LiDARChart />
        </div>
      </div>

      {/* Full width history table */}
      <div className="w-full">
        <LiDARHistoryTable />
      </div>
    </div>
  );
};

export default LiDARData;
