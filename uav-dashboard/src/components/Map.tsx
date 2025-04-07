import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Marker,
  Popup,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";

const Map = () => {
  const [path, setPath] = useState<[number, number, number][]>([]);
  interface Obstacle {
    position: [number, number, number];
    shape: string;
  }

  const [obstacles, setObstacles] = useState<Obstacle[]>([]);

  useEffect(() => {
    // Fetch the UAV path and obstacles from the backend
    fetch("http://localhost:8000/api/simulation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: [0, 0, 0],
        destination: [15, 15, 8],
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        setPath(data.best_path);
        setObstacles(data.obstacles);
      })
      .catch((error) =>
        console.error("Error fetching simulation data:", error)
      );
  }, []);

  return (
    <div className="h-[400px] rounded-lg overflow-hidden shadow-lg">
      <MapContainer
        center={[10, 10]}
        zoom={10}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {/* Display the UAV path */}
        <Polyline
          positions={path.map(([x, y]) => [y, x])} // Convert to [lat, lon]
          color="blue"
        />
        {/* Display obstacles */}
        {obstacles.map((obs, index) => (
          <Marker
            key={index}
            position={[obs.position[1], obs.position[0]]} // Convert to [lat, lon]
          >
            <Popup>Obstacle: {obs.shape}</Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};

export default Map;
