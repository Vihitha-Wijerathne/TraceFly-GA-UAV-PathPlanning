from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Survivor
from app.schemas import SurvivorSchema

router = APIRouter()

@router.post("/")
def store_survivor_data(data: SurvivorSchema, db: Session = Depends(get_db)):
    survivor = Survivor(**data.dict())
    db.add(survivor)
    db.commit()
    db.refresh(survivor)
    return survivor

@router.get("/{uav_id}")
def get_survivor_data(uav_id: str, db: Session = Depends(get_db)):
    survivors = db.query(Survivor).filter(Survivor.uav_id == uav_id).all()
    if not survivors:
        return {"message": "No survivors found"}
    return survivors
