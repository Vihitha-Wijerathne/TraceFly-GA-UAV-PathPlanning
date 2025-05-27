from fastapi import APIRouter
import requests

router = APIRouter()

@router.post("/start")
def start_simulation():
    try:
        r = requests.post("http://localhost:8081/start")
        return {"status": "started", "unity_status": r.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/stop")
def stop_simulation():
    try:
        r = requests.post("http://localhost:8081/stop")
        return {"status": "stopped", "unity_status": r.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/pause")
def pause_simulation():
    try:
        r = requests.post("http://localhost:8081/pause")
        return {"status": "paused", "unity_status": r.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.post("/resume")
def resume_simulation():
    try:
        r = requests.post("http://localhost:8081/resume")
        return {"status": "resumed", "unity_status": r.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}
