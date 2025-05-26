from fastapi import APIRouter

router = APIRouter()

@router.get("/api/bandwidth")
def get_bandwidth_mode():
    return {"mode": "low"}