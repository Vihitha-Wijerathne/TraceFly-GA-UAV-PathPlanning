from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel
from typing import List
from datetime import datetime
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import LidarObstacle, LiDARRecord
from app.schemas import LidarObstacleSchema
from app.schemas import LiDARPingLite, LiDARPayload, Point3D

router = APIRouter()

LIDAR_HIT_HISTORY = {}

class LiDARTestPing(BaseModel):
    message: str
    
class LiDARPointCloud(BaseModel):
    uav_id: str
    timestamp: datetime
    points: List[List[float]]  # [x, y, z] triples
    
class LiDARPingLite(BaseModel):
    uav_id: str
    timestamp: datetime
    hit_count: int
    
@router.post("/")
def store_lidar_data(data: LidarObstacleSchema, db: Session = Depends(get_db)):
    obstacle = LidarObstacle(**data.dict())
    db.add(obstacle)
    db.commit()
    db.refresh(obstacle)
    return obstacle

# @router.get("/unity/history/{uav_id}")
# def get_lidar_history(uav_id: str, db: Session = Depends(get_db)):
#     records = db.query(LiDARRecord).filter(LiDARRecord.uav_id == uav_id).order_by(LiDARRecord.timestamp).all()
#     return [{"timestamp": r.timestamp, "hit_count": r.hit_count} for r in records]


# @router.post("/unity")
# def receive_lidar_minimal(payload: LiDARPingLite):
#     print(f"[LiDAR ✅] {payload.uav_id} sent {payload.hit_count} hits at {payload.timestamp}")
#     return {"status": "ok", "received_hits": payload.hit_count}

# @router.post("/unity/full")
# def store_pointcloud(payload: LiDARPointCloud, db: Session = Depends(get_db)):
#     record = LiDARRecord(
#         uav_id=payload.uav_id,
#         timestamp=payload.timestamp,
#         hit_count=len(payload.points),
#         points=payload.points  
#     )
#     db.add(record)
#     db.commit()
#     return {"status": "stored", "points": len(payload.points)}

@router.post("/unity")
def receive_lidar_minimal(payload: LiDARPingLite, db: Session = Depends(get_db)):
    print(f"[LiDAR ✅] {payload.uav_id} sent {payload.hit_count} hits at {payload.timestamp}")

    record = LiDARRecord(
        uav_id=payload.uav_id,
        timestamp=payload.timestamp,
        hit_count=payload.hit_count
    )
    db.add(record)
    db.commit()
    return {"status": "ok", "received_hits": payload.hit_count}

@router.get("/unity/history/{uav_id}")
def get_lidar_history(uav_id: str):
    # Return the latest 100 records, including tags!
    hlist = LIDAR_HIT_HISTORY.get(uav_id, [])
    result = []
    for entry in hlist:
        result.append({
            "timestamp": entry["timestamp"],
            "hit_count": entry["hit_count"],
            "tags": [hit.tag for hit in entry["hits"]],
        })
    return result
@router.post("/unity/full")
def store_pointcloud(payload: LiDARPayload):
    # Keep last 100 for each UAV
    hlist = LIDAR_HIT_HISTORY.setdefault(payload.uav_id, [])
    entry = {
        "timestamp": payload.timestamp,
        "hits": payload.hits,
        "hit_count": len(payload.hits),
    }
    hlist.append(entry)
    # Keep only latest 100
    if len(hlist) > 100:
        hlist.pop(0)
    return {"status": "stored", "count": len(payload.hits)}