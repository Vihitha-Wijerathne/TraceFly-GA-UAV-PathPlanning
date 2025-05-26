from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from .GA_complex_env import ComplexEnvironment
from .GA_enhanced import GeneticAlgorithm

router = APIRouter()

class SimulationRequest(BaseModel):
    start: List[float]  # [x, y, z]
    destination: List[float]

@router.post("/start")
def start_simulation(request: SimulationRequest):
    try:
        x_bounds = (0, 100)
        y_bounds = (0, 50)
        z_bounds = (0, 100)
        num_obstacles = 20

        environment = ComplexEnvironment(x_bounds, y_bounds, z_bounds, num_obstacles)
        ga = GeneticAlgorithm(population_size=50, generations=100, mutation_rate=0.2)

        best_path = ga.run(tuple(request.start), tuple(request.destination), environment)

        return {
            "best_path": best_path,
            "obstacles": environment.obstacles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
