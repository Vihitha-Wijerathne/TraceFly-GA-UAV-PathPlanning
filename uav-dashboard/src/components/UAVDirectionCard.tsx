import { useEffect, useState } from "react";

declare global {
  interface Window {
    prevLat?: number;
    prevLon?: number;
    prevAlt?: number;
  }
}

const UAVDirectionCard = () => {
  const [movement, setMovement] = useState({
    up: false,
    down: false,
    left: false,
    right: false,
    forward: false,
    backward: false,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://localhost:8000/api/telemetry/unity/latest")
        .then((res) => res.json())
        .then((data) => {
          const dx = data.latitude - (window.prevLat || data.latitude);
          const dz = data.longitude - (window.prevLon || data.longitude);
          const dy = data.altitude - (window.prevAlt || data.altitude);

          setMovement({
            up: dy > 0.5,
            down: dy < -0.5,
            forward: dx > 0.5,
            backward: dx < -0.5,
            left: dz < -0.5,
            right: dz > 0.5,
          });

          window.prevLat = data.latitude;
          window.prevLon = data.longitude;
          window.prevAlt = data.altitude;
        });
    }, 500);

    return () => clearInterval(interval);
  }, []);

  const active = "bg-blue-500 text-white shadow-md";
  const base =
    "w-24 h-24 flex items-center justify-center border rounded-full transition duration-300 ease-in-out text-lg font-semibold";

  return (
    <div className="bg-white p-6 rounded-xl shadow-md space-y-4 w-full max-w-sm mx-auto">
      <h3 className="text-lg font-semibold text-gray-800 text-center">
        UAV Movement Indicator
      </h3>
      <div className="flex justify-center">
        <div className={`${base} ${movement.up ? active : "text-gray-600"}`}>
          ↑
        </div>
      </div>
      <div className="flex justify-between items-center space-x-6 px-4">
        <div className={`${base} ${movement.left ? active : "text-gray-600"}`}>
          ←
        </div>
        <div
          className={`${base} ${movement.forward ? active : "text-gray-600"}`}
        >
          ⮝
        </div>
        <div className={`${base} ${movement.right ? active : "text-gray-600"}`}>
          →
        </div>
      </div>
      <div className="flex justify-center">
        <div className={`${base} ${movement.down ? active : "text-gray-600"}`}>
          ↓
        </div>
      </div>
      <div className="flex justify-center">
        <div
          className={`${base} ${movement.backward ? active : "text-gray-600"}`}
        >
          ⮟
        </div>
      </div>
    </div>
  );
};

export default UAVDirectionCard;
