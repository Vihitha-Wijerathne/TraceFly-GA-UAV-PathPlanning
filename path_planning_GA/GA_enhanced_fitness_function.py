import numpy as np
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
        obstacles = []
        for _ in range(self.num_obstacles):
            x = np.random.randint(*self.x_bounds)
            y = np.random.randint(*self.y_bounds)
            z = np.random.randint(*self.z_bounds)
            obstacles.append((x, y, z))
        return obstacles

    def is_obstacle(self, point):
        return point in self.obstacles

    def distance_to_obstacles(self, point):
        distances = [np.linalg.norm(np.array(point) - np.array(obs)) for obs in self.obstacles]
        return min(distances) if distances else float('inf')

# Path planner for initial paths
class PathPlanner:
    def __init__(self, source, destination, environment):
        self.source = source
        self.destination = destination
        self.environment = environment

    def find_path(self, dodge_strategy):
        current = self.source
        path = [current]

        while current != self.destination:
            next_point = self.find_next_step(current, self.destination)

            if self.environment.is_obstacle(next_point):
                current = self.dodge_obstacle(current, dodge_strategy)
            else:
                current = next_point
            
            if current in path:
                break  # Prevent infinite loops
            path.append(current)

        return path

    def find_next_step(self, current, target):
        direction = (
            np.sign(target[0] - current[0]),
            np.sign(target[1] - current[1]),
            np.sign(target[2] - current[2]),
        )
        return (current[0] + direction[0], current[1] + direction[1], current[2] + direction[2])

    def dodge_obstacle(self, current, dodge_strategy):
        moves = {
            'up': (0, 0, 1),
            'down': (0, 0, -1),
            'left': (-1, 0, 0),
            'right': (1, 0, 0),
            'forward': (0, 1, 0),
            'backward': (0, -1, 0),
        }
        move = moves[dodge_strategy]
        next_point = (current[0] + move[0], current[1] + move[1], current[2] + move[2])
        return next_point if self.environment.is_within_bounds(next_point) else current

# Genetic Algorithm for path optimization
class GeneticAlgorithm:
    def __init__(self, population, environment, generations=50, mutation_rate=0.1):
        self.population = population
        self.environment = environment
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_path = None

    def calculate_fitness(self, path):
        obstacle_penalty = sum(1 for point in path if self.environment.is_obstacle(point))
        safety_score = 1 / (1 + obstacle_penalty)

        turns = sum(1 for i in range(1, len(path) - 1) if path[i - 1] != path[i + 1])
        altitude_changes = sum(abs(path[i][2] - path[i - 1][2]) for i in range(1, len(path)))
        energy_score = 1 / (1 + turns + altitude_changes)

        distance = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
        time_score = 1 / distance

        return 0.5 * safety_score + 0.3 * energy_score + 0.2 * time_score

    def select_parents(self):
        fitness_scores = [self.calculate_fitness(path) for path in self.population]
        probabilities = [score / sum(fitness_scores) for score in fitness_scores]
        parents_indices = np.random.choice(len(self.population), size=2, p=probabilities, replace=False)
        return self.population[parents_indices[0]], self.population[parents_indices[1]]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, min(len(parent1), len(parent2)) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, path):
        if np.random.rand() < self.mutation_rate:
            index = np.random.randint(1, len(path) - 1)
            path[index] = (path[index][0] + np.random.choice([-1, 1]),
                           path[index][1] + np.random.choice([-1, 1]),
                           path[index][2] + np.random.choice([-1, 1]))
        return path

    def evolve(self):
        for _ in range(self.generations):
            new_population = []
            for _ in range(len(self.population)):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population

        self.best_path = max(self.population, key=self.calculate_fitness)

# Visualization
def plot_paths(paths, best_path, environment, source, destination):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles
    obstacles = np.array(environment.obstacles)
    ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c='red', label='Obstacles')

    # Plot source and destination
    ax.scatter(*source, c='blue', s=100, label='Source')
    ax.scatter(*destination, c='purple', s=100, label='Destination')

    # Plot all paths
    for path in paths:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], label='Initial Path', alpha=0.4)

    # Plot the best path
    best_path_array = np.array(best_path)
    ax.plot(best_path_array[:, 0], best_path_array[:, 1], best_path_array[:, 2], c='green', linewidth=2, label='Best Path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


# Main script
x_bounds = (0, 20)
y_bounds = (0, 20)
z_bounds = (0, 10)
source = (0, 0, 0)
destination = (20, 20, 10)

# Initialize environment
environment = Environment(x_bounds, y_bounds, z_bounds, num_obstacles=400)

# Generate initial paths
planner = PathPlanner(source, destination, environment)
dodge_strategies = ['up', 'down', 'left', 'right', 'forward', 'backward']
initial_paths = [planner.find_path(strategy) for strategy in dodge_strategies]

# Run Genetic Algorithm
ga = GeneticAlgorithm(initial_paths, environment, generations=100, mutation_rate=0.2)
ga.evolve()

# Visualize paths and the best path
plot_paths(initial_paths, ga.best_path, environment, source, destination)
