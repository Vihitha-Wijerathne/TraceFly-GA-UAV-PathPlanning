from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .GA_complex_env import ComplexEnvironment
from .GA_complex_PF import GeneticAlgorithm
router = APIRouter()

class SimulationRequest(BaseModel):
    start: tuple[int, int, int]
    destination: tuple[int, int, int]

@router.post("/start")
def start_simulation(request: SimulationRequest):
    try:
        # Define the environment bounds and obstacles
        x_bounds = (0, 20)
        y_bounds = (0, 20)
        z_bounds = (0, 10)
        num_obstacles = 10

        # Initialize the environment
        environment = ComplexEnvironment(x_bounds, y_bounds, z_bounds, num_obstacles)

        # Initialize the Genetic Algorithm
        ga = GeneticAlgorithm(
            population_size=50,
            mutation_rate=0.1,
            generations=100,
            environment=environment,
            source=request.start,
            destination=request.destination,
        )

        # Run the simulation
        ga.evolve()
        best_path = ga.best_path()

        # Return the best path
        return {
            "best_path": [point for point in best_path.waypoints],
            "obstacles": environment.obstacles,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))