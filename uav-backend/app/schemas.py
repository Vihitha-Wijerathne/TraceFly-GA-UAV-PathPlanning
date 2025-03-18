from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

# Telemetry Schema
class TelemetrySchema(BaseModel):
    uav_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    altitude: float
    imu_acc_x: float
    imu_acc_y: float
    imu_acc_z: float
    imu_gyro_x: float
    imu_gyro_y: float
    imu_gyro_z: float
    speed: float
    wind_speed: float
    battery_level: float

    class Config:
        orm_mode = True

# Path Planning Schema
class PathPlanningSchema(BaseModel):
    uav_id: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    waypoints: List[List[float]]  # List of [lat, lon]

    class Config:
        orm_mode = True

# LiDAR Obstacle Schema
class LidarObstacleSchema(BaseModel):
    uav_id: str
    obstacle_lat: float
    obstacle_lon: float
    obstacle_height: float
    detected_at: Optional[datetime] = None

    class Config:
        orm_mode = True

# Survivor Schema
class SurvivorSchema(BaseModel):
    uav_id: str
    survivor_lat: float
    survivor_lon: float
    condition: str
    detected_at: Optional[datetime] = None

    class Config:
        orm_mode = True
