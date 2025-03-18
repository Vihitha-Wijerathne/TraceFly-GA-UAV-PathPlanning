from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Telemetry
from app.schemas import TelemetrySchema

router = APIRouter()

@router.post("/")
def store_telemetry(data: TelemetrySchema, db: Session = Depends(get_db)):
    telemetry = Telemetry(**data.dict())
    db.add(telemetry)
    db.commit()
    db.refresh(telemetry)
    return telemetry

@router.get("/{uav_id}")
def get_telemetry(uav_id: str, db: Session = Depends(get_db)):
    telemetry = db.query(Telemetry).filter(Telemetry.uav_id == uav_id).order_by(Telemetry.timestamp.desc()).first()
    if not telemetry:
        return {"message": "No telemetry found"}
    return telemetry
