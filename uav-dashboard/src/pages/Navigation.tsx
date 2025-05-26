import UAVDirectionCard from "../components/UAVDirectionCard";

const Navigation = () => {
  return (
    <div>
      <div className="container mx-auto p-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <UAVDirectionCard />
        </div>
      </div>
    </div>
  );
};

export default Navigation;
