import React, { useEffect, useState } from "react";
import { useTelemetry } from "../hooks/useTelemetry";
import { useNavigate } from "react-router-dom";
import {
  BatteryFull,
  Ruler,
  Compass,
  Wind,
  GaugeCircle,
  WifiIcon,
  Navigation,
  AlertTriangle,
  AlertCircle,
} from "lucide-react";
import TelemetryCard from "../components/TelemetryCard";

// Types
type TelemetryData = {
  uav_id: string;
  latitude: number;
  longitude: number;
  altitude: number;
  pitch: number;
  roll: number;
  yaw: number;
  speed: number;
  battery_level: number;
  wind: number;
  signal: string;
};

// Helper for Yaw → Compass
const getDirectionFromYaw = (yaw: number): string => {
  if (yaw >= 337.5 || yaw < 22.5) return "N";
  if (yaw >= 22.5 && yaw < 67.5) return "NE";
  if (yaw >= 67.5 && yaw < 112.5) return "E";
  if (yaw >= 112.5 && yaw < 157.5) return "SE";
  if (yaw >= 157.5 && yaw < 202.5) return "S";
  if (yaw >= 202.5 && yaw < 247.5) return "SW";
  if (yaw >= 247.5 && yaw < 292.5) return "W";
  if (yaw >= 292.5 && yaw < 337.5) return "NW";
  return "N/A";
};

// Helper for obstacle status label & style
const getObstacleLabel = (hitCount: number) => {
  if (hitCount <= 3)
    return { label: "Clear", style: "bg-green-100 text-green-700" };
  if (hitCount <= 12)
    return { label: "Few Obstacles", style: "bg-yellow-100 text-yellow-800" };
  return { label: "Heavy Obstacles", style: "bg-red-100 text-red-700" };
};

const TelemetryPanel = () => {
  const data = useTelemetry() as TelemetryData | null;
  const navigate = useNavigate();

  // LiDAR hit count state
  const [lidarHits, setLidarHits] = useState(0);
  const [showStopPrompt, setShowStopPrompt] = useState(false);
  const [criticalPrompt, setCriticalPrompt] = useState(false);

  // Fetch latest LiDAR hit count
  useEffect(() => {
    const fetchLatestHits = async () => {
      try {
        const res = await fetch(
          "http://localhost:8000/api/lidar/unity/history/drone_test"
        );
        const history = await res.json();
        if (Array.isArray(history) && history.length > 0) {
          const last = history[history.length - 1];
          setLidarHits(last.hit_count);
          setShowStopPrompt(last.hit_count > 21);
        } else {
          setLidarHits(0);
          setShowStopPrompt(false);
        }
      } catch {
        setLidarHits(0);
        setShowStopPrompt(false);
      }
    };
    fetchLatestHits();
    const interval = setInterval(fetchLatestHits, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleStop = async () => {
    await fetch("http://localhost:8000/api/simulation/stop", {
      method: "POST",
    });
    setShowStopPrompt(false);
    setCriticalPrompt(false);
  };

  const handleResume = async () => {
    await fetch("http://localhost:8000/api/simulation/resume", {
      method: "POST",
    });
    setCriticalPrompt(false);
  };

  // Watch signal for low bandwidth changes and close prompt when out of low zone
  useEffect(() => {
    if (data && data.signal === "low") {
      setCriticalPrompt(true);
    } else {
      setCriticalPrompt(false);
    }
  }, [data?.signal]);

  const obstacle = getObstacleLabel(lidarHits);

  if (!data)
    return (
      <div className="text-gray-500 text-center mt-6">Loading telemetry...</div>
    );

  const {
    latitude,
    longitude,
    altitude,
    pitch,
    roll,
    yaw,
    speed,
    battery_level,
    wind,
    signal,
  } = data;

  const batteryColor =
    battery_level >= 50
      ? "text-green-500"
      : battery_level >= 20
      ? "text-yellow-500"
      : "text-red-500";

  const signalColor =
    signal === "high"
      ? "text-green-600"
      : signal === "medium"
      ? "text-yellow-500"
      : "text-red-600";

  const direction = getDirectionFromYaw(yaw);

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-xl font-semibold text-gray-800">📡 Drone Status</h2>
      {/* --- Bandwidth Warnings --- */}
      {signal === "medium" && (
        <div className="flex items-center gap-2 bg-orange-50 border-l-4 border-orange-400 text-orange-700 p-3 rounded mb-4">
          <AlertTriangle className="w-5 h-5 text-orange-500" />
          <span className="font-semibold">Warning:</span>
          UAV is currently in a <b>limited bandwidth (Medium Signal)</b> area!
        </div>
      )}
      {/* --- Critical Bandwidth Prompt --- */}
      {criticalPrompt && signal === "low" && (
        <div className="flex flex-col items-center bg-red-50 border-l-4 border-red-600 text-red-700 p-4 rounded mb-4">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-6 h-6 text-red-600" />
            <span className="font-semibold">Critical:</span>
            UAV is now in a <b>critically low bandwidth</b> zone!
          </div>
          <div className="mt-4 flex gap-4">
            <button
              className="px-4 py-2 bg-green-600 text-white rounded font-bold hover:bg-green-700 transition"
              onClick={handleResume}
            >
              RESUME
            </button>
            <button
              className="px-4 py-2 bg-red-600 text-white rounded font-bold hover:bg-red-700 transition"
              onClick={async () => {
                await handleStop();
                navigate("/"); // Go to dashboard
                setTimeout(() => {
                  alert("Please enter a new destination to fly.");
                }, 500);
              }}
            >
              STOP
            </button>
          </div>
        </div>
      )}

      {/* --- Main Telemetry Cards --- */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        <TelemetryCard
          label="Altitude"
          value={altitude?.toFixed(1) || "N/A"}
          unit="m"
          icon={<Ruler className="text-blue-600" />}
        />
        <TelemetryCard
          label="Longitude"
          value={longitude?.toFixed(5) || "N/A"}
          icon={<Ruler className="text-blue-600" />}
        />
        <TelemetryCard
          label="Latitude"
          value={latitude?.toFixed(5) || "N/A"}
          icon={<Ruler className="text-blue-600" />}
        />
        <TelemetryCard
          label="Battery"
          value={`${battery_level?.toFixed(0)}%`}
          icon={<BatteryFull className={batteryColor} />}
          color={batteryColor}
        />
        <TelemetryCard
          label="Speed"
          value={speed?.toFixed(2) || "N/A"}
          unit="m/s"
          icon={<GaugeCircle className="text-indigo-600" />}
        />
        <TelemetryCard
          label="Pitch"
          value={pitch?.toFixed(1) || "N/A"}
          unit="°"
          icon={<Compass className="text-purple-500" />}
        />
        <TelemetryCard
          label="Roll"
          value={roll?.toFixed(1) || "N/A"}
          unit="°"
          icon={<Compass className="text-purple-500" />}
        />
        <TelemetryCard
          label="Yaw"
          value={yaw?.toFixed(1) || "N/A"}
          unit="°"
          icon={<Compass className="text-purple-500" />}
        />
        <TelemetryCard
          label="Wind"
          value={wind?.toFixed(1) || "N/A"}
          unit="m/s"
          icon={<Wind className="text-cyan-500" />}
        />
        {/* <TelemetryCard
          label="Signal"
          value={signal?.toUpperCase() || "N/A"}
          icon={<WifiIcon className={signalColor} />}
          color={signalColor}
        /> */}
        <TelemetryCard
          label="Heading"
          value={direction}
          icon={<Navigation className="text-orange-500" />}
        />
        {/* ----- Obstacle Card ----- */}
        <div className="bg-white p-4 rounded-lg shadow text-sm w-full flex justify-between items-center">
          <div>
            <div className="text-gray-500 mb-1">Obstacle</div>
            <div
              className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${obstacle.style}`}
            >
              {obstacle.label}
            </div>
          </div>
        </div>
      </div>
      {/* ----- STOP Prompt if Heavy Obstacles -----
      {showStopPrompt && (
        <div className="my-6 bg-red-50 border border-red-200 p-4 rounded shadow flex flex-col items-center">
          <div className="text-red-700 text-md font-semibold mb-2">
            The drone is in a obstacle full environment, want to stop the drone?
          </div>
          <button
            className="px-4 py-2 bg-red-600 text-white rounded font-bold hover:bg-red-700 transition"
            onClick={handleStop}
          >
            STOP
          </button>
        </div>
      )} */}
    </div>
  );
};

export default TelemetryPanel;
