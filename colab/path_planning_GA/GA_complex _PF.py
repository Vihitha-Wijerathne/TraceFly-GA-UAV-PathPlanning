import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from random import choice


# Complex Environment with Obstacles
class ComplexEnvironment:
    def __init__(self, x_bounds, y_bounds, z_bounds, num_obstacles):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.num_obstacles = num_obstacles
        self.obstacles = self.generate_complex_obstacles()

    def generate_complex_obstacles(self):
        obstacles = []
        for _ in range(self.num_obstacles):
            shape = choice(["cube", "sphere", "cylinder"])
            x = np.random.randint(*self.x_bounds)
            y = np.random.randint(*self.y_bounds)
            z = np.random.randint(*self.z_bounds)
            size = np.random.randint(1, 5)  # Size of the obstacle
            obstacles.append({"shape": shape, "position": (x, y, z), "size": size})
        return obstacles

    def is_obstacle(self, point):
        for obs in self.obstacles:
            shape = obs["shape"]
            position = obs["position"]
            size = obs["size"]
            if shape == "cube":
                if all(abs(point[i] - position[i]) <= size for i in range(3)):
                    return True
            elif shape == "sphere":
                if np.linalg.norm(np.array(point) - np.array(position)) <= size:
                    return True
            elif shape == "cylinder":
                if (abs(point[0] - position[0]) <= size and
                        abs(point[1] - position[1]) <= size and
                        abs(point[2] - position[2]) <= 2 * size):
                    return True
        return False

    def distance_to_obstacles(self, point):
        distances = []
        for obs in self.obstacles:
            shape = obs["shape"]
            position = obs["position"]
            size = obs["size"]
            if shape == "cube":
                dist = max(0, np.linalg.norm(np.array(point) - np.array(position)) - size)
            elif shape == "sphere":
                dist = max(0, np.linalg.norm(np.array(point) - np.array(position)) - size)
            elif shape == "cylinder":
                dist = max(0, abs(point[2] - position[2]) - 2 * size)
            distances.append(dist)
        return min(distances) if distances else float('inf')

    def plot_environment(self, ax):
        for obs in self.obstacles:
            shape = obs["shape"]
            position = obs["position"]
            size = obs["size"]
            if shape == "cube":
                # Draw cube
                x, y, z = position
                r = [-size, size]
                verts = [[(x + dx, y + dy, z + dz) for dx in r for dy in r for dz in r]]
                ax.add_collection3d(Poly3DCollection(verts, alpha=0.5, color="gray"))
            elif shape == "sphere":
                # Draw sphere
                u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                x = size * np.cos(u) * np.sin(v) + position[0]
                y = size * np.sin(u) * np.sin(v) + position[1]
                z = size * np.cos(v) + position[2]
                ax.plot_surface(x, y, z, color="gray", alpha=0.5)
            elif shape == "cylinder":
                # Draw cylinder
                x = np.linspace(-size, size, 50) + position[0]
                z = np.linspace(0, 2 * size, 50) + position[2]
                X, Z = np.meshgrid(x, z)
                Y = np.sqrt(size**2 - X**2) + position[1]
                ax.plot_surface(X, Y, Z, color="gray", alpha=0.5)


# Path Planner with Height Variations
class PathPlanner:
    def __init__(self, source, destination, environment):
        self.source = source
        self.destination = destination
        self.environment = environment

    def find_path(self, dodge_strategy, height_variation=0):
        current = self.source
        path = [current]

        while current != self.destination:
            next_point = self.find_next_step(current, self.destination, height_variation)

            if self.environment.is_obstacle(next_point):
                current = self.dodge_obstacle(current, dodge_strategy)
            else:
                current = next_point

            if current in path:
                break  # Prevent infinite loops
            path.append(current)

        return path

    def find_next_step(self, current, target, height_variation):
        direction = (
            np.sign(target[0] - current[0]),
            np.sign(target[1] - current[1]),
            np.sign(target[2] - current[2] + height_variation),
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
        return next_point if not self.environment.is_obstacle(next_point) else current


# Fitness Function with Multiple Objectives
class Path:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.fitness = 0

    def is_turning(self, i):
        return self.waypoints[i][0] != self.waypoints[i - 1][0] or self.waypoints[i][1] != self.waypoints[i - 1][1]

    def calculate_fitness(self, environment, wind_data=None):
        # Safety Score
        obstacle_penalty = sum(1 for point in self.waypoints if environment.is_obstacle(point)) + \
                           sum(1 / (1 + environment.distance_to_obstacles(point)) for point in self.waypoints)
        safety_score = 1 / (1 + obstacle_penalty)

        # Energy Score
        turns = sum(1 for i in range(1, len(self.waypoints) - 1) if self.is_turning(i))
        altitude_changes = sum(abs(self.waypoints[i][2] - self.waypoints[i-1][2]) for i in range(1, len(self.waypoints)))
        energy_penalty = turns + altitude_changes
        energy_score = 1 / (1 + energy_penalty)

        # Height-based penalties for wind/turbulence
        height_penalty = sum(abs(self.waypoints[i][2] - self.waypoints[i-1][2]) for i in range(1, len(self.waypoints)))
        wind_score = 1 / (1 + height_penalty)

        # Time Score (distance-based)
        distance = sum(np.linalg.norm(np.array(self.waypoints[i]) - np.array(self.waypoints[i + 1]))
                       for i in range(len(self.waypoints) - 1))
        time_score = 1 / distance

        # Combine the scores(final fitness values)
        self.fitness = (0.4 * safety_score +
                        0.2 * energy_score +
                        0.2 * wind_score +
                        0.1 * time_score +
                        0.1 * (1 / (1 + turns)))


# Genetic Algorithm for Evolving Paths
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, environment, source, destination):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.environment = environment
        self.source = source
        self.destination = destination
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            path_planner = PathPlanner(self.source, self.destination, self.environment)
            path = path_planner.find_path(dodge_strategy='up', height_variation=np.random.randint(-2, 2))
            population.append(Path(path))
        return population

    def select_parents(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[:2]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1.waypoints) - 1)
        child1 = parent1.waypoints[:crossover_point] + parent2.waypoints[crossover_point:]
        child2 = parent2.waypoints[:crossover_point] + parent1.waypoints[crossover_point:]
        return Path(child1), Path(child2)

    def mutate(self, child):
        if np.random.rand() < self.mutation_rate:
            mutation_point = np.random.randint(0, len(child.waypoints))
            child.waypoints[mutation_point] = (child.waypoints[mutation_point][0] + np.random.choice([-1, 1]),
                                                child.waypoints[mutation_point][1] + np.random.choice([-1, 1]),
                                                child.waypoints[mutation_point][2] + np.random.choice([-1, 1]))
        return child

    def evolve(self):
        for generation in range(self.generations):
            parents = self.select_parents()
            child1, child2 = self.crossover(parents[0], parents[1])
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            self.population.append(child1)
            self.population.append(child2)
            self.population = self.population[:self.population_size]
            for path in self.population:
                path.calculate_fitness(self.environment)

    def best_path(self):
        return max(self.population, key=lambda x: x.fitness)


# Initialize Environment and Run Genetic Algorithm 
if __name__ == '__main__':
    x_bounds = (0, 20)
    y_bounds = (0, 20)
    z_bounds = (0, 10)
    num_obstacles = 10

    environment = ComplexEnvironment(x_bounds, y_bounds, z_bounds, num_obstacles)
    source = (0, 0, 0)
    destination = (15, 15, 8)

    ga = GeneticAlgorithm(population_size=50, mutation_rate=0.1, generations=100, environment=environment,
                          source=source, destination=destination)
    ga.evolve()

    best_path = ga.best_path()

    # Plot the environment and best path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    environment.plot_environment(ax)
    path_x, path_y, path_z = zip(*best_path.waypoints)
    ax.plot(path_x, path_y, path_z, label='Best Path', color='r')
    ax.scatter(*source, color='g', label='Source')
    ax.scatter(*destination, color='b', label='Destination')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()
