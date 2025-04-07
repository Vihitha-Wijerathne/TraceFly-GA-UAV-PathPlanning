import { Battery, Compass, Gauge, Ruler } from "lucide-react";
import Map from "../components/Map";
import TelemetryChart from "../components/TelemetryChart";
import { useTelemetry } from "../context/TelemetryContext";
import PathSelection from "../components/PathSelection";

const Dashboard = () => {
  const { telemetryData } = useTelemetry();

  const stats = [
    {
      label: "Altitude",
      value: `${telemetryData.altitude.toFixed(1)}m`,
      icon: Ruler,
      color: "text-blue-600",
    },
    {
      label: "Speed",
      value: `${telemetryData.speed.toFixed(1)}m/s`,
      icon: Gauge,
      color: "text-green-600",
    },
    {
      label: "Battery",
      value: `${telemetryData.battery.toFixed(1)}%`,
      icon: Battery,
      color: "text-yellow-600",
    },
    {
      label: "Heading",
      value: `${telemetryData.heading.toFixed(1)}°`,
      icon: Compass,
      color: "text-purple-600",
    },
  ];
  const handlePathSubmit = (
    start: [number, number, number],
    destination: [number, number, number]
  ) => {
    console.log("Start:", start, "Destination:", destination);
    // Send the start and destination to the backend
    fetch("http://localhost:8000/api/simulation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start, destination }),
    })
      .then((response) => response.json())
      .then((data) => console.log("Simulation started:", data))
      .catch((error) => console.error("Error:", error));
  };

  return (
    <div>
      <div className="container mx-auto p-6 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {stats.map((stat, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">{stat.label}</p>
                  <p className={`text-2xl font-semibold ${stat.color}`}>
                    {stat.value}
                  </p>
                </div>
                <stat.icon className={`w-8 h-8 ${stat.color}`} />
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Map />
          <TelemetryChart />
        </div>
      </div>
      <div className="container mx-auto p-4">
        <h1 className="text-2xl font-bold mb-6">UAV Path Planning Dashboard</h1>
        <PathSelection onSubmit={handlePathSubmit} />
      </div>
    </div>
  );
};

export default Dashboard;
