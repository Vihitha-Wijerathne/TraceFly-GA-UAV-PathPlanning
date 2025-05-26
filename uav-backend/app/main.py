from fastapi import FastAPI, WebSocket
from app.routers import telemetry, path_planning, lidar, survivors
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from app.database import Base, engine

app = FastAPI(title="UAV Disaster Response API")

Base.metadata.create_all(bind=engine)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../colab")))

# Allow all origins for WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register Routers
app.include_router(telemetry.router, prefix="/api/telemetry", tags=["Telemetry"])
app.include_router(path_planning.router, prefix="/api/path-planning", tags=["Path Planning"])
app.include_router(lidar.router, prefix="/api/lidar", tags=["LiDAR"])
app.include_router(survivors.router, prefix="/api/survivors", tags=["Survivors"])
# app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# WebSocket Test Route
@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Hello from WebSocket!")
    await websocket.close()