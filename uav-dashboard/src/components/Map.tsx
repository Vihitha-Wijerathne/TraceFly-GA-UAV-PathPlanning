import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import { useTelemetry } from '../context/TelemetryContext';
import 'leaflet/dist/leaflet.css';

const Map = () => {
  const { telemetryData } = useTelemetry();
  const position: [number, number] = [telemetryData.latitude, telemetryData.longitude];

  return (
    <div className="h-[400px] rounded-lg overflow-hidden shadow-lg">
      <MapContainer
        center={position}
        zoom={13}
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <Marker position={position}>
          <Popup>
            Altitude: {telemetryData.altitude.toFixed(2)}m<br />
            Speed: {telemetryData.speed.toFixed(2)}m/s
          </Popup>
        </Marker>
      </MapContainer>
    </div>
  );
};

export default Map;