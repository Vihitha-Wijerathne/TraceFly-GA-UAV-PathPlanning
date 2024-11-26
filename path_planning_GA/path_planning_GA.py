import numpy as np
import random
import matplotlib.pyplot as plt

#Define parameters 
POPULATION_SIZE = 200
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

        #Calculate noise score
        noise_penalty = turns + (altitude_changes / 10)
        noise_score = 1 / (1 + noise_penalty)
        
        #calculate smoothness score
        smoothness_penalty = turns #Simple penalty for turns
        smoothness_score = 1 / (1 + smoothness_penalty)

        #Combine the scores using weighted formula
        self.fitness = (0.5 * safety_score +
                        0.2 * energy_score +
                        0.2 * time_score +
                        0.05 * noise_score +
                        0.05 * smoothness_score)
        
#generate initial population
def gen_population(start, destination, size):
    population = []
    for _ in range(size):
        waypoints = [start] + [random_point() for _ in range(random.randint(1, 5))] + [destination]
        population.append(Path(waypoints))
    return population

def random_point():
    return (random.randint(0, env.width), random.randint(0, env.height))

#GA operations
#Selection
def selection(population):
    population.sort(key = lambda path: path.fitness, reverse = True)
    return population[:POPULATION_SIZE // 2]

#Crossover
def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1.waypoints) - 1)
    child_waypoints = parent1.waypoints[:cut] + parent2.waypoints[cut:]
    return Path(child_waypoints)

#Mutate
def mutate(path):
    if random.random() < MUTATION_RATE:
        index = random.randint(1, len(path.waypoints) - 2)
        path.waypoints[index] = random_point()

#Main Algorithm
env = Enviroment(width = 100, height = 100, obstacles = [(50, 50), (51, 51), (52, 50)])
start = (0, 0)
destination = (99, 99)

#Run GA
population = gen_population(start, destination, POPULATION_SIZE)

for generation in range(GENERATIONS):
    for path in population:
        path.calculate_fitness(destination)

    selected = selection(population)
    next_generation = selected.copy()

    while len(next_generation) < POPULATION_SIZE:
        parent1, parent2 = random.sample(selected, 2)
        child = crossover(parent1, parent2)
        mutate(child)
        next_generation.append(child)
        
    population = next_generation

#get the best path
best_path = max(population, key=lambda path: path.fitness)

#visualization
plt.figure()
plt.xlim(0, env.width)
plt.ylim(0, env.height)
plt.plot(*zip(*best_path.waypoints), marker='o')
plt.scatter(*zip(*env.obsatcles), color = 'red')
plt.scatter(start[0], start[1], color = 'green')
plt.scatter(destination[0], destination[1], color='blue')
plt.title("Best path found by the algorithm")
plt.show()