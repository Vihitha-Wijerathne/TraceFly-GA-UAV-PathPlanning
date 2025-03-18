import React from 'react';
import { Battery, Compass, Gauge, Ruler } from 'lucide-react';
import Map from '../components/Map';
import TelemetryChart from '../components/TelemetryChart';
import { useTelemetry } from '../context/TelemetryContext';

const Dashboard = () => {
  const { telemetryData } = useTelemetry();

  const stats = [
    {
      label: 'Altitude',
      value: `${telemetryData.altitude.toFixed(1)}m`,
      icon: Ruler,
      color: 'text-blue-600'
    },
    {
      label: 'Speed',
      value: `${telemetryData.speed.toFixed(1)}m/s`,
      icon: Gauge,
      color: 'text-green-600'
    },
    {
      label: 'Battery',
      value: `${telemetryData.battery.toFixed(1)}%`,
      icon: Battery,
      color: 'text-yellow-600'
    },
    {
      label: 'Heading',
      value: `${telemetryData.heading.toFixed(1)}°`,
      icon: Compass,
      color: 'text-purple-600'
    }
  ];

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <div key={index} className="bg-white p-6 rounded-lg shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">{stat.label}</p>
                <p className={`text-2xl font-semibold ${stat.color}`}>{stat.value}</p>
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
  );
};

export default Dashboard;