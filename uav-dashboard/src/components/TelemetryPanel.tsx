import { useTelemetry } from "../hooks/useTelemetry";

type TelemetryData = {
  uav_id: string;
  latitude: number;
  longitude: number;
  altitude: number;
  pitch : number;
  roll: number;
  yaw: number;
  speed: number;
  battery_level: number;
  wind: number;
  signal: string;
};

const TelemetryPanel = () => {
  const data = useTelemetry() as TelemetryData | null;

  if (!data) return <div>Loading...</div>;

  const {
    uav_id,
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

  return (
    <div className="bg-gray-800 text-white p-4 rounded">
      <h2 className="text-lg font-bold mb-2">📡 UAV Telemetry</h2>
      <div>UAV ID: {uav_id ?? "N/A"}</div>
      <div>Lat: {latitude !== undefined ? latitude.toFixed(5) : "N/A"}</div>
      <div>Lon: {longitude !== undefined ? longitude.toFixed(5) : "N/A"}</div>
      <div>Alt: {altitude !== undefined ? altitude.toFixed(2) : "N/A"} m</div>
      <div>Pitch: {pitch !== undefined ? pitch.toFixed(1) : "N/A"} m/s</div>
      <div>Roll: {roll !== undefined ? roll.toFixed(1) : "N/A"} m/s</div>
      <div>Yaw: {yaw !== undefined ? yaw.toFixed(1) : "N/A"} m/s</div>
      <div>Speed: {speed !== undefined ? speed.toFixed(1) : "N/A"} m/s</div>
      <div>
        Battery:{" "}
        {battery_level !== undefined ? battery_level.toFixed(0) : "N/A"}%
      </div>
      <div>Wind: {wind !== undefined ? wind.toFixed(1) : "N/A"} m/s</div>
      <div>Signal: {signal?.toUpperCase() ?? "N/A"}</div>
    </div>
  );
};

export default TelemetryPanel;
