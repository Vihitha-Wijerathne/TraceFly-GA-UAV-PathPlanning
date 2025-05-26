import numpy as np  
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Telemetry
from app.services.telemetry_service import compress_gps, compress_imu
from app.schemas import TelemetrySchema
from fastapi import WebSocket, WebSocketDisconnect
from typing import List
from app.schemas import UnityTelemetrySchema

router = APIRouter()
active_connections: List[WebSocket] = []
latest_telemetry_data = {}

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

    Returns a dictionary with all NumPy types converted to native Python types.
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
        prioritized_indices = np.argsort(-signal_strength)  # Descending order
        prioritized_data = {
            "latitude": latitude[prioritized_indices].tolist(),
            "longitude": longitude[prioritized_indices].tolist(),
            "signal_strength": signal_strength[prioritized_indices].tolist(),
        }

        return {
            "compressed_lat": compressed_lat.decode("latin1"),
            "compressed_lon": compressed_lon.decode("latin1"),
            "imu_rle_values": imu_rle_values.tolist() if hasattr(imu_rle_values, "tolist") else imu_rle_values,
            "imu_rle_counts": imu_rle_counts.tolist() if hasattr(imu_rle_counts, "tolist") else imu_rle_counts,
            "imu_quant_min": int(imu_quant_min) if isinstance(imu_quant_min, np.generic) else imu_quant_min,
            "imu_quant_step": int(imu_quant_step) if isinstance(imu_quant_step, np.generic) else imu_quant_step,
            "prioritized_data": prioritized_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/gps")
def receive_gps(data: dict = Body(...)):
    """
    Receives compressed GPS data from UAV.
    Example:
    {
        "uav_id": "drone01",
        "timestamp": "2025-05-25T14:31:00",
        "lat": 123.456789,
        "lon": 78.910111,
        "alt": 90.0
    }
    """
    print(f"[GPS] {data['uav_id']} @ {data['lat']:.6f}, {data['lon']:.6f}, alt={data['alt']:.1f}")
    return {"status": "received", "type": "gps"}


@router.post("/imu")
def receive_imu(data: dict = Body(...)):
    """
    Receives compressed IMU data from UAV.
    Example:
    {
        "uav_id": "drone01",
        "timestamp": "2025-05-25T14:31:00",
        "ax": 0.1, "ay": 0.0, "az": 9.8,
        "yaw": 45.2, "pitch": 3.4, "roll": 2.1
    }
    """
    print(f"[IMU] {data['uav_id']} @ acc=({data['ax']}, {data['ay']}, {data['az']}) | angles=({data['pitch']}, {data['roll']}, {data['yaw']})")
    return {"status": "received", "type": "imu"}


@router.post("/battery")
def receive_battery(data: dict = Body(...)):
    """
    Receives battery/signal status from UAV.
    Example:
    {
        "uav_id": "drone01",
        "timestamp": "2025-05-25T14:31:00",
        "battery": 88.5,
        "signal": "medium"
    }
    """
    print(f"[Battery] {data['uav_id']} battery={data['battery']}%, signal={data['signal']}")
    return {"status": "received", "type": "battery"}


@router.post("/lidar")
def receive_lidar(data: dict = Body(...)):
    """
    Receives LiDAR hit data (optional).
    Example:
    {
        "uav_id": "drone01",
        "timestamp": "2025-05-25T14:31:00",
        "lidar_hits": [
            {"x": 1.0, "y": 2.0, "z": 3.0},
            ...
        ]
    }
    """
    print(f"[LiDAR] {data['uav_id']} hits={len(data.get('lidar_hits', []))}")
    return {"status": "received", "type": "lidar"}

@router.post("/unity")
def receive_unity_telemetry(payload: UnityTelemetrySchema, db: Session = Depends(get_db)):
    print(f"[Unity POST] {payload.uav_id} @ {payload.lat}, {payload.lon}, alt={payload.alt}")
    global latest_telemetry_data

    latest_telemetry_data = {
        "uav_id": payload.uav_id,
        "timestamp": payload.timestamp,
        "latitude": payload.lat,
        "longitude": payload.lon,
        "altitude": payload.alt,
        "pitch": payload.pitch,
        "roll": payload.roll,
        "yaw": payload.yaw,
        "speed": payload.speed,
        "battery_level": payload.battery,
        "wind": payload.wind,
        "signal_strength": payload.signal,
    }

    return {"message": "Unity telemetry received"}


@router.get("/unity/latest")
def get_latest_telemetry():
    return latest_telemetry_data