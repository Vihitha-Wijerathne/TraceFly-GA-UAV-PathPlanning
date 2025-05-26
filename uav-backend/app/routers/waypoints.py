from fastapi import APIRouter

router = APIRouter()

@router.get("/api/waypoints")
def get_waypoints():
    waypoints = [
        {"x": 0, "y": 1.0, "z": 0},
        {"x": 5, "y": 2.5, "z": 10},
        {"x": 10, "y": 1.5, "z": 20},
    ]
    return {"waypoints": waypoints}