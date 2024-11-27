import numpy as np
import random
import matplotlib.pyplot as plt

# Environment class with larger and strategically placed obstacles
class Environment:
    def __init__(self, x_bounds, y_bounds, z_bounds, num_random_obstacles=30):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.num_random_obstacles = num_random_obstacles
        self.obstacles = self.generate_obstacles()

    def generate_obstacles(self):
        obstacles = []

        # Add 2 large obstacles in the middle of the environment
        mid_x = (self.x_bounds[0] + self.x_bounds[1]) // 2
        mid_y = (self.y_bounds[0] + self.y_bounds[1]) // 2
        mid_z = (self.z_bounds[0] + self.z_bounds[1]) // 2

        # First large obstacle (cuboid)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-1, 2):  # Shorter in height
                    obstacles.append((mid_x + dx, mid_y + dy, mid_z + dz))

        # Second large obstacle (cuboid)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    obstacles.append((mid_x + dx + 5, mid_y + dy - 5, mid_z + dz))

        # Add random obstacles scattered around the environment
        for _ in range(self.num_random_obstacles):
            x = np.random.randint(*self.x_bounds)
            y = np.random.randint(*self.y_bounds)
            z = np.random.randint(*self.z_bounds)
            obstacles.append((x, y, z))

        return list(set(obstacles))  # Remove duplicates

# UAV Path Planner for individual path generation
class UAVPathPlanner:
    def __init__(self, source, destination, environment):
        self.source = source
        self.destination = destination
        self.environment = environment
        self.path = [self.source]

    def is_within_bounds(self, point):
        x, y, z = point
        return (
            self.environment.x_bounds[0] <= x <= self.environment.x_bounds[1]
            and self.environment.y_bounds[0] <= y <= self.environment.y_bounds[1]
            and self.environment.z_bounds[0] <= z <= self.environment.z_bounds[1]
        )

    def is_collision(self, point):
        return point in self.environment.obstacles

    def compute_straight_line(self, start, end):
        points = []
        num_steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), abs(end[2] - start[2]))
        for t in np.linspace(0, 1, num_steps + 1):
            x = int(round(start[0] + t * (end[0] - start[0])))
            y = int(round(start[1] + t * (end[1] - start[1])))
            z = int(round(start[2] + t * (end[2] - start[2])))
            points.append((x, y, z))
        return points

    def dodge_obstacle(self, current, target):
        possible_moves = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
        random.shuffle(possible_moves)  # Add randomness for diversity
        for move in possible_moves:
            next_point = (current[0] + move[0], current[1] + move[1], current[2] + move[2])
            if self.is_within_bounds(next_point) and not self.is_collision(next_point):
                return next_point
        return current  # Edge case: no move possible

    def find_path(self):
        current = self.source
        while current != self.destination:
            straight_line = self.compute_straight_line(current, self.destination)
            for point in straight_line:
                if self.is_collision(point):
                    current = self.dodge_obstacle(current, self.destination)
                    break
                else:
                    self.path.append(point)
                    current = point
                    if current == self.destination:
                        break
        return self.path

# Genetic Algorithm for optimizing paths
class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []

    def initialize_population(self, source, destination, environment):
        # Generate initial population using UAVPathPlanner
        for _ in range(self.population_size):
            planner = UAVPathPlanner(source, destination, environment)
            path = planner.find_path()
            self.population.append(path)

    def fitness_function(self, path):
        # Fitness based on path length and straightness
        path_length = len(path)
        straightness_penalty = sum(
            1 for i in range(1, len(path) - 1) if path[i - 1] != path[i + 1]
        )
        return -path_length - straightness_penalty

    def selection(self):
        # Roulette wheel selection
        fitness_scores = [self.fitness_function(path) for path in self.population]
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = random.choices(self.population, probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        # Single-point crossover
        split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2

    def mutate(self, path):
        # Randomly modify a waypoint in the path
        if len(path) > 2 and random.random() < self.mutation_rate:
            mutate_idx = random.randint(1, len(path) - 2)
            planner = UAVPathPlanner(path[mutate_idx - 1], path[-1], environment)
            path[mutate_idx] = planner.dodge_obstacle(path[mutate_idx - 1], path[-1])

    def evolve(self):
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self.selection()
            child1, child2 = self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            new_population.extend([child1, child2])
        self.population = new_population

    def run(self, source, destination, environment):
        self.initialize_population(source, destination, environment)
        for generation in range(self.generations):
            self.evolve()
            best_path = max(self.population, key=self.fitness_function)
            print(f"Generation {generation + 1}, Best Fitness: {self.fitness_function(best_path)}")
        return best_path

    def plot_paths(self, environment, best_path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot obstacles
        obstacles = np.array(environment.obstacles)
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c='red', label='Obstacles')

        # Plot best path
        best_path = np.array(best_path)
        ax.plot(best_path[:, 0], best_path[:, 1], best_path[:, 2], c='green', linewidth=2, label='Best Path')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()

# Sample Input
x_bounds = (0, 30)
y_bounds = (0, 30)
z_bounds = (0, 15)
source = (0, 0, 0)
destination = (30, 30, 15)

# Generate environment
environment = Environment(x_bounds, y_bounds, z_bounds)

# Run GA
ga = GeneticAlgorithm(population_size=20, generations=30, mutation_rate=0.2)
best_path = ga.run(source, destination, environment)
ga.plot_paths(environment, best_path)