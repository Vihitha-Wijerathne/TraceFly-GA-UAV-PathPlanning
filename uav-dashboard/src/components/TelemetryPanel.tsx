import { useTelemetry } from "../hooks/useTelemetry";
import {
  BatteryFull,
  Ruler,
  Compass,
  Wind,
  GaugeCircle,
  WifiIcon,
  Navigation,
} from "lucide-react";
import TelemetryCard from "../components/TelemetryCard";

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

const TelemetryPanel = () => {
  const data = useTelemetry() as TelemetryData | null;

  if (!data)
    return <div className="text-gray-500 text-center mt-6">Loading telemetry...</div>;

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
      </div>
    </div>
  );
};

export default TelemetryPanel;
