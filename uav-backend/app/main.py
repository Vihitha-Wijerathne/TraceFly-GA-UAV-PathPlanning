from fastapi import FastAPI
from app.routers import telemetry, path_planning, lidar, survivors

app = FastAPI(title="UAV Disaster Response API")

# Register Routers
app.include_router(telemetry.router, prefix="/api/telemetry", tags=["Telemetry"])
app.include_router(path_planning.router, prefix="/api/path-planning", tags=["Path Planning"])
app.include_router(lidar.router, prefix="/api/lidar", tags=["LiDAR"])
app.include_router(survivors.router, prefix="/api/survivors", tags=["Survivors"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
