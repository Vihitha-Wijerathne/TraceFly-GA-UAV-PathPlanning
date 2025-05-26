import React from "react";
import { Link } from "react-router-dom";
import { Bone as Drone, History, Settings } from "lucide-react";
import { useTelemetry } from "../context/TelemetryContext";

const Header = () => {
  const { isConnected } = useTelemetry();

  return (
    <header className="bg-slate-800 text-white p-4">
      <div className="container mx-auto flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-2">
          <Drone size={24} />
          <span className="text-xl font-bold">TraceFly</span>
        </Link>

        <nav className="flex items-center space-x-6">
          {/* <Link
            to="/"
            className="flex items-center space-x-1 hover:text-blue-400"
          >
            <Drone size={18} />
            <span>Dashboard</span>
          </Link> */}
          <Link
            to="/uavmap"
            className="flex items-center space-x-1 hover:text-blue-400"
          >
            <Drone size={18} />
            <span>LiDAR</span>
          </Link>
          <Link
            to="/navigation"
            className="flex items-center space-x-1 hover:text-blue-400"
          >
            <Drone size={18} />
            <span>UAV Navigation</span>
          </Link>
          <Link
            to="/telemetryPanel"
            className="flex items-center space-x-1 hover:text-blue-400"
          >
            <Drone size={18} />
            <span>Telemetry</span>
          </Link>
          <Link
            to="/history"
            className="flex items-center space-x-1 hover:text-blue-400"
          >
            <History size={18} />
            <span>History</span>
          </Link>
          <Link
            to="/settings"
            className="flex items-center space-x-1 hover:text-blue-400"
          >
            <Settings size={18} />
            <span>Settings</span>
          </Link>
          <div className="flex items-center space-x-2">
            <div
              className={`w-3 h-3 rounded-full ${
                isConnected ? "bg-green-500" : "bg-red-500"
              }`}
            ></div>
            <span className="text-sm">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;
