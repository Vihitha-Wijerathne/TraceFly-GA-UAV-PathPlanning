import LiDARChart from "../components/LiDARChart";

const LiDARData = () => {

  return (
    <div>
      <div className="container mx-auto p-6 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <LiDARChart />
        </div>
      </div>
    </div>
  );
};

export default LiDARData;
