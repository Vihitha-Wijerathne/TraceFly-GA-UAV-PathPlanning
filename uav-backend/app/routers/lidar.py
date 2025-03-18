from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import LidarObstacle
from app.schemas import LidarObstacleSchema

router = APIRouter()

@router.post("/")
def store_lidar_data(data: LidarObstacleSchema, db: Session = Depends(get_db)):
    obstacle = LidarObstacle(**data.dict())
    db.add(obstacle)
    db.commit()
    db.refresh(obstacle)
    return obstacle

@router.get("/{uav_id}")
def get_lidar_data(uav_id: str, db: Session = Depends(get_db)):
    obstacles = db.query(LidarObstacle).filter(LidarObstacle.uav_id == uav_id).all()
    if not obstacles:
        return {"message": "No LiDAR obstacles found"}
    return obstacles
