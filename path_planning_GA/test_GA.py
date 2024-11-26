import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the environment
class Environment:
    def __init__(self, x_bounds, y_bounds, z_bounds, num_obstacles):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.num_obstacles = num_obstacles
        self.obstacles = self.generate_obstacles()

    def generate_obstacles(self):
        # Randomly generate obstacle coordinates within bounds
        obstacles = []
        for _ in range(self.num_obstacles):
            x = random.randint(*self.x_bounds)
            y = random.randint(*self.y_bounds)
            z = random.randint(*self.z_bounds)
            obstacles.append((x, y, z))
        return obstacles
    
# Define the UAV path planner with Genetic Algorithm
class UAVPathPlannerGA:
    def __init__(self, source, destination, environment, population_size=50, generations=100):
        self.source = source
        self.destination = destination
        self.environment = environment
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_path = None

    def initialize_population(self):
        # Create random paths as initial population
        for _ in range(self.population_size):
            path = [self.source]
            current = self.source
            while current != self.destination:
                next_point = self.generate_next_point(current)
                path.append(next_point)
                current = next_point
            self.population.append(path)