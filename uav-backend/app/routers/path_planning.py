from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import PathPlanning
from app.schemas import PathPlanningSchema

router = APIRouter()

@router.post("/")
def create_path(path_data: PathPlanningSchema, db: Session = Depends(get_db)):
    new_path = PathPlanning(
        uav_id=path_data.uav_id,
        start_lat=path_data.start_lat,
        start_lon=path_data.start_lon,
        end_lat=path_data.end_lat,
        end_lon=path_data.end_lon,
        waypoints=str(path_data.waypoints)
    )
    db.add(new_path)
    db.commit()
    db.refresh(new_path)
    return new_path

@router.get("/{uav_id}")
def get_path(uav_id: str, db: Session = Depends(get_db)):
    path = db.query(PathPlanning).filter(PathPlanning.uav_id == uav_id).first()
    if not path:
        return {"message": "Path not found"}
    return path
