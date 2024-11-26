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
    def generate_next_point(self, current):
        # Generate next point considering movements in 3D space
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        while True:
            move = random.choice(directions)
            next_point = (current[0] + move[0], current[1] + move[1], current[2] + move[2])
            if self.is_within_bounds(next_point) and next_point not in self.environment.obstacles:
                return next_point

    def is_within_bounds(self, point):
        x, y, z = point
        return (
            self.environment.x_bounds[0] <= x <= self.environment.x_bounds[1]
            and self.environment.y_bounds[0] <= y <= self.environment.y_bounds[1]
            and self.environment.z_bounds[0] <= z <= self.environment.z_bounds[1]
        )

    def fitness_function(self, path):
        # Fitness function based on path length and obstacle avoidance
        path_length = len(path)
        collisions = sum(1 for point in path if point in self.environment.obstacles)
        straightness = self.calculate_straightness(path)
        return -path_length - (collisions * 10) - straightness

    def calculate_straightness(self, path):
        # Penalize paths with excessive direction changes
        changes = 0
        for i in range(1, len(path) - 1):
            if path[i - 1] != path[i + 1]:
                changes += 1
        return changes

    def crossover(self, parent1, parent2):
        # Single-point crossover
        split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2

    def mutate(self, path):
        # Randomly modify part of the path
        if len(path) > 2:
            mutate_idx = random.randint(1, len(path) - 2)
            path[mutate_idx] = self.generate_next_point(path[mutate_idx - 1])

    def evolve(self):
        # Perform selection, crossover, and mutation
        new_population = []
        fitness_scores = [self.fitness_function(path) for path in self.population]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), key=lambda x: -x[0])]
        self.best_path = sorted_population[0]

        for i in range(0, self.population_size, 2):
            parent1, parent2 = sorted_population[i], sorted_population[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])

        self.population = new_population

    def run(self):
        self.initialize_population()
        for generation in range(self.generations):
            self.evolve()
            print(f"Generation {generation + 1}: Best fitness = {self.fitness_function(self.best_path)}")

    def plot_paths(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot obstacles
        obstacles = np.array(self.environment.obstacles)
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c='red', label='Obstacles')

        # Plot alternative paths
        for path in self.population:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], linestyle='dashed', alpha=0.5, label='Alternative Path')

        # Plot best path
        best_path = np.array(self.best_path)
        ax.plot(best_path[:, 0], best_path[:, 1], best_path[:, 2], c='green', linewidth=2, label='Best Path')

        # Plot source and destination
        ax.scatter(*self.source, c='blue', s=100, label='Source')
        ax.scatter(*self.destination, c='purple', s=100, label='Destination')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()


# Sample Input
x_bounds = (0, 20)
y_bounds = (0, 20)
z_bounds = (0, 10)
source = (0, 0, 0)
destination = (20, 20, 10)

# Generate environment with obstacles
environment = Environment(x_bounds, y_bounds, z_bounds, num_obstacles=50)

# Run the UAV Path Planner
planner = UAVPathPlannerGA(source, destination, environment)
planner.run()
planner.plot_paths()