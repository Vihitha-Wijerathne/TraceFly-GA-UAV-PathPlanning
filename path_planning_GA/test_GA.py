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