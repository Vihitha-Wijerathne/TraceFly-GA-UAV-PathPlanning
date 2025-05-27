import React, { useEffect, useState } from "react";
import UAVDirectionCard from "../components/UAVDirectionCard";

const useHeavyObstaclePrompt = () => {
  const [showPrompt, setShowPrompt] = useState(false);
  useEffect(() => {
    const fetchHits = async () => {
      try {
        const res = await fetch(
          "http://localhost:8000/api/lidar/unity/history/drone_test"
        );
        const history = await res.json();
        if (Array.isArray(history) && history.length > 0) {
          const last = history[history.length - 1];
          setShowPrompt(last.hit_count > 21);
        } else {
          setShowPrompt(false);
        }
      } catch {
        setShowPrompt(false);
      }
    };
    fetchHits();
    const interval = setInterval(fetchHits, 1000);
    return () => clearInterval(interval);
  }, []);
  return showPrompt;
};

const Navigation = () => {
  const showHeavyObstaclePrompt = useHeavyObstaclePrompt();

  const backDrone = async () => {
    await fetch("http://localhost:8000/api/simulation/stop", {
      method: "POST",
    });
  };
  const pauseDrone = async () => {
    await fetch("http://localhost:8000/api/simulation/pause", {
      method: "POST",
    });
  };
  const resumeDrone = async () => {
    await fetch("http://localhost:8000/api/simulation/resume", {
      method: "POST",
    });
  };

  return (
    <div className="min-h-screen bg-gray-100 py-10">
      <div className="container mx-auto p-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <UAVDirectionCard />
        </div>

        {/* Heavy obstacle prompt */}
        {showHeavyObstaclePrompt && (
          <div className="my-6 bg-red-50 border border-red-200 p-4 rounded shadow flex flex-col items-center">
            <div className="text-red-700 text-md font-semibold mb-2">
              The drone is in an obstacle-full environment. Do you want to make
              the UAV go back to the starting point?
            </div>
            <button
              className="px-8 py-3 rounded-lg bg-white text-red-600 border border-red-200 font-semibold shadow hover:bg-red-50 hover:border-red-400 transition focus:outline-none focus:ring-2 focus:ring-red-200"
              onClick={backDrone}
            >
              BACK
            </button>
          </div>
        )}

        <div className="flex gap-4 mt-10 justify-center">
          <button
            className="px-8 py-3 rounded-lg bg-white text-red-600 border border-red-200 font-semibold shadow hover:bg-red-50 hover:border-red-400 transition focus:outline-none focus:ring-2 focus:ring-red-200"
            onClick={backDrone}
          >
            BACK
          </button>
          <button
            className="px-8 py-3 rounded-lg bg-white text-yellow-700 border border-yellow-200 font-semibold shadow hover:bg-yellow-50 hover:border-yellow-400 transition focus:outline-none focus:ring-2 focus:ring-yellow-200"
            onClick={pauseDrone}
          >
            PAUSE
          </button>
          <button
            className="px-8 py-3 rounded-lg bg-white text-green-600 border border-green-200 font-semibold shadow hover:bg-green-50 hover:border-green-400 transition focus:outline-none focus:ring-2 focus:ring-green-200"
            onClick={resumeDrone}
          >
            RESUME
          </button>
        </div>
      </div>
    </div>
  );
};

export default Navigation;
