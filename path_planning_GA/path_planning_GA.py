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
    
#Define Path:
class Path:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.fitness = 0

    def is_turning(self, index):
        #check if the path is making a turn
        if index < 1 or index >= len(self.waypoints) - 1:
            return False
        p1 = np.array(self.waypoints[index - 1])
        p2 = np.array(self.waypoints[index])
        p3 = np.array(self.waypoints[index + 1])
        angle = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        return abs(angle) > np.pi / 6 #threshold for a turn

    def calculate_fitness(self, destination):
        #Calculate safety score
        obstacle_penalty = sum(1 for point in self.waypoints if env.is_obstacle(point)) + \
                           sum(1 / (1 + env.distance_to_obstacles(point)) for point in self.waypoints)
        safety_score = 1 / (1 + obstacle_penalty)

        #Calculate energy score
        turns = sum(1 for i in range(1, len(self.waypoints) - 1)
                    if self.is_turning(i))
        altitude_changes = sum(abs(self.waypoints[i][1] - self.waypoints[i-1][1]) for i in range(1, len(self.waypoints)))
        energy_penalty = turns + altitude_changes
        energy_score = 1 / (1 + energy_penalty)

        #Calculate time score
        distance = sum(np.linalg.norm(np.array(self.waypoints[i]) - np.array(self.waypoints[i + 1]))
                       for i in range(len(self.waypoints) - 1))
        time_score = 1 / distance
        