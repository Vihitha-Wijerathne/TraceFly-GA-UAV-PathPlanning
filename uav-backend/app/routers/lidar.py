from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import LidarObstacle
from app.schemas import LidarObstacleSchema
import torch.nn as nn 

router = APIRouter()

class PointPillars(nn.Module):
    def __init__(self, voxel_size, point_cloud_range):
        super(PointPillars, self).__init__()
        self.voxel_layer = Voxelization(voxel_size, point_cloud_range)  # type: ignore
        self.cnn_encoder = PillarFeatureNet()  # type: ignore
    
    def forward(self, points):
        voxels = self.voxel_layer(points)
        features = self.cnn_encoder(voxels)
        return features  # 512-dimensional vector

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
