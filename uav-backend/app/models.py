from sqlalchemy import Column, Integer, String, Float, DateTime, func
from app.database import Base

class Telemetry(Base):
    __tablename__ = 'telemetry'
    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)
    speed = Column(Float, nullable=False)
    heading = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=func.now())

class PathPlanning(Base):
    __tablename__ = 'path_planning'
    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(String, nullable=False)
    start_lat = Column(Float, nullable=False)
    start_lon = Column(Float, nullable=False)
    end_lat = Column(Float, nullable=False)
    end_lon = Column(Float, nullable=False)
    waypoints = Column(String, nullable=False)  # JSON string of waypoints

class LidarObstacle(Base):
    __tablename__ = 'lidar_obstacles'
    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(String, nullable=False)
    obstacle_lat = Column(Float, nullable=False)
    obstacle_lon = Column(Float, nullable=False)
    obstacle_height = Column(Float, nullable=False)
    detected_at = Column(DateTime, default=func.now())

class Survivor(Base):
    __tablename__ = 'survivors'
    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(String, nullable=False)
    survivor_lat = Column(Float, nullable=False)
    survivor_lon = Column(Float, nullable=False)
    condition = Column(String, nullable=False)  # e.g., "critical", "stable"
    detected_at = Column(DateTime, default=func.now())