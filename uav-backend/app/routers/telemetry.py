import numpy as np  
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Telemetry
from app.services.telemetry_service import compress_gps, compress_imu
from app.schemas import TelemetrySchema
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

router = APIRouter()
active_connections: List[WebSocket] = []

@router.post("/", response_model=TelemetrySchema)
def upload_telemetry(data: TelemetrySchema, db: Session = Depends(get_db)):
    compressed_lat, compressed_lon = compress_gps(
        [data.latitude], [data.longitude]
    )

    # Convert IMU data to NumPy array before passing it to compress_imu
    imu_data = np.array([
        data.imu_acc_x, data.imu_acc_y, data.imu_acc_z,
        data.imu_gyro_x, data.imu_gyro_y, data.imu_gyro_z
    ])

    imu_rle_values, imu_rle_counts, imu_quant_min, imu_quant_step = compress_imu(imu_data)

    # Ensure imu_rle_values and imu_rle_counts are NumPy arrays before calling tobytes()
    imu_rle_values = np.array(imu_rle_values)
    imu_rle_counts = np.array(imu_rle_counts)

    telemetry_entry = Telemetry(
        uav_id=data.uav_id,
        timestamp=data.timestamp,
        latitude=data.latitude,
        longitude=data.longitude,
        altitude=data.altitude,
        imu_acc_x=data.imu_acc_x,
        imu_acc_y=data.imu_acc_y,
        imu_acc_z=data.imu_acc_z,
        imu_gyro_x=data.imu_gyro_x,
        imu_gyro_y=data.imu_gyro_y,
        imu_gyro_z=data.imu_gyro_z,
        speed=data.speed,
        wind_speed=data.wind_speed,
        battery_level=data.battery_level,
        delta_lat=compressed_lat,
        delta_lon=compressed_lon,
        imu_rle_values=imu_rle_values.tobytes(),  # Convert to bytes
        imu_rle_counts=imu_rle_counts.tobytes(),  # Convert to bytes
        imu_quant_min=float(imu_quant_min),  # Convert to standard Python float
        imu_quant_step=float(imu_quant_step)  # Convert to standard Python float
    )

    db.add(telemetry_entry)
    db.commit()
    db.refresh(telemetry_entry)
    return telemetry_entry

@router.websocket("/ws/telemetry/{uav_id}")
async def telemetry_stream(websocket: WebSocket, uav_id: str):
    """WebSocket connection for UAV telemetry streaming."""
    print(f"🔄 Attempting WebSocket connection for UAV: {uav_id}")

    try:
        # Accept WebSocket connection
        await websocket.accept()
        active_connections.append(websocket)
        print(f"✅ WebSocket connected for UAV: {uav_id}")

        while True:
            # Receive data
            data = await websocket.receive_json()
            print(f"📡 Received data: {data}")

            # Send data to all active WebSocket connections
            for connection in active_connections:
                await connection.send_json(data)

    except WebSocketDisconnect:
        print(f"⚠️ WebSocket disconnected for UAV: {uav_id}")
        active_connections.remove(websocket)
        
@router.post("/process")
def process_telemetry(data: dict):
    """
    Processes telemetry data by compressing and prioritizing it.

    Parameters:
    - data: A dictionary containing telemetry data (e.g., GPS, IMU).

    Returns:
    - A dictionary with compressed and prioritized telemetry data.
    """
    try:
        # Extract GPS and IMU data
        latitude = np.array(data.get("latitude", []))
        longitude = np.array(data.get("longitude", []))
        imu_data = np.array(data.get("imu", []))
        signal_strength = np.array(data.get("signal_strength", []))

        # Compress GPS data
        compressed_lat, compressed_lon = compress_gps(latitude, longitude)

        # Compress IMU data
        imu_rle_values, imu_rle_counts, imu_quant_min, imu_quant_step = compress_imu(imu_data)

        # Prioritize data based on signal strength
        prioritized_indices = np.argsort(-signal_strength)  # Sort in descending order
        prioritized_data = {
            "latitude": latitude[prioritized_indices].tolist(),
            "longitude": longitude[prioritized_indices].tolist(),
            "signal_strength": signal_strength[prioritized_indices].tolist(),
        }

        return {
            "compressed_lat": compressed_lat.decode("latin1"),  # Convert bytes to string for JSON
            "compressed_lon": compressed_lon.decode("latin1"),
            "imu_rle_values": imu_rle_values,
            "imu_rle_counts": imu_rle_counts,
            "imu_quant_min": imu_quant_min,
            "imu_quant_step": imu_quant_step,
            "prioritized_data": prioritized_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))