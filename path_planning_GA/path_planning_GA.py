import numpy as np
import random
import matplotlib.pyplot as plt

#Define parameters 
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
GENERATIONS = 50

#Define the enviroment
class Enviroment:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obsatcles = obstacles #List of (x, y) tuples

    def is_obstacle(self, point):
        return point in self.obsatcles
    
    def distance_to_obstacles(self, point):
        return min(np.linalg.norm(np.array(point) - np.array(obs)) for obs in self.obsatcles)
    

        