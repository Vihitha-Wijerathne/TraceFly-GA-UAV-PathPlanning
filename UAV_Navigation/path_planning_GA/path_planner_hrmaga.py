import numpy as np
import json
import os
from typing import List, Tuple
from datetime import datetime

# ========================
# Utility: Read Obstacle Data
# ========================
def load_obstacles_from_json(filepath: str) -> List[dict]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("detected_objects", [])

def is_collision(point, obstacles, safety_radius=2.0):
    for obs in obstacles:
        pos = np.array(obs["position"])
        size = obs.get("size", 1.0)
        if np.linalg.norm(np.array(point) - pos) <= size + safety_radius:
            return True
    return False

def distance_to_nearest_obstacle(point, obstacles):
    if not obstacles:
        return float('inf')
    return min(np.linalg.norm(np.array(point) - np.array(obs["position"])) for obs in obstacles)

# ========================
# Genetic Algorithm Components
# ========================
class Path:
    def __init__(self, waypoints: List[Tuple[int, int, int]]):
        self.waypoints = waypoints
        self.fitness = 0

    def is_turning(self, i):
        return (self.waypoints[i][0] != self.waypoints[i - 1][0] or
                self.waypoints[i][1] != self.waypoints[i - 1][1])

    def calculate_fitness(self, obstacles):
        turns = sum(1 for i in range(1, len(self.waypoints) - 1) if self.is_turning(i))
        altitude_changes = sum(abs(self.waypoints[i][2] - self.waypoints[i - 1][2]) for i in range(1, len(self.waypoints)))
        obstacle_penalty = sum(1 for pt in self.waypoints if is_collision(pt, obstacles)) + \
                           sum(1 / (1 + distance_to_nearest_obstacle(pt, obstacles)) for pt in self.waypoints)
        safety_score = 1 / (1 + obstacle_penalty)
        energy_score = 1 / (1 + turns + altitude_changes)
        wind_score = 1 / (1 + altitude_changes)

        distance = sum(np.linalg.norm(np.array(self.waypoints[i]) - np.array(self.waypoints[i + 1]))
                       for i in range(len(self.waypoints) - 1))
        time_score = 1 / distance if distance != 0 else 0

        self.fitness = (0.4 * safety_score +
                        0.2 * energy_score +
                        0.2 * wind_score +
                        0.1 * time_score +
                        0.1 * (1 / (1 + turns)))

class HRMAGAGenerator:
    def __init__(self, population_size=50, mutation_rate=0.1, generations=100):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.emergency_stop = False

    def generate_initial_path(self, start, end):
        path = [start]
        current = np.array(start)

        while not np.array_equal(current, end):
            step = np.clip(np.array(end) - current, -1, 1)
            next_step = tuple(current + step)
            if next_step == path[-1]:
                break
            path.append(next_step)
            current = np.array(next_step)

        return path

    def initialize_population(self, start, end, obstacles):
        return [Path(self.generate_initial_path(start, end)) for _ in range(self.population_size)]

    def crossover(self, p1, p2):
        cp = np.random.randint(1, min(len(p1.waypoints), len(p2.waypoints)) - 1)
        child1_wp = p1.waypoints[:cp] + p2.waypoints[cp:]
        child2_wp = p2.waypoints[:cp] + p1.waypoints[cp:]
        return Path(child1_wp), Path(child2_wp)

    def mutate(self, path):
        if len(path.waypoints) == 0:
            return path
        if np.random.rand() < self.mutation_rate:
            mp = np.random.randint(1, len(path.waypoints) - 1)
            dx, dy, dz = np.random.choice([-1, 0, 1], 3)
            mutated = tuple(np.array(path.waypoints[mp]) + [dx, dy, dz])
            path.waypoints[mp] = mutated
        return path

    def evolve(self, start, end, obstacles):
        population = self.initialize_population(start, end, obstacles)
        for path in population:
            path.calculate_fitness(obstacles)

        best_path = None
        best_fitness = -1

        for gen in range(self.generations):
            if self.emergency_stop:
                print("[⚠️ EMERGENCY HALT TRIGGERED] Path planning interrupted.")
                return []

            population.sort(key=lambda p: p.fitness, reverse=True)
            next_gen = population[:2]

            while len(next_gen) < self.population_size:
                p1, p2 = np.random.choice(population[:10], 2, replace=False)
                c1, c2 = self.crossover(p1, p2)
                next_gen.extend([self.mutate(c1), self.mutate(c2)])

            population = next_gen[:self.population_size]
            for path in population:
                path.calculate_fitness(obstacles)

            if population[0].fitness > best_fitness:
                best_fitness = population[0].fitness
                best_path = population[0]

        return best_path.waypoints

    def emergency_halt(self):
        self.emergency_stop = True

# ========================
# Wrapper Method to Run HR-MAGA for Waypoints
# ========================
def generate_local_path(start_point, end_point):
    lidar_file = "data/detected_objects.json"
    output_file = "data/ga_path_output.json"
    os.makedirs("data", exist_ok=True)

    obstacles = load_obstacles_from_json(lidar_file)
    planner = HRMAGAGenerator()
    best_path = planner.evolve(start_point, end_point, obstacles)

    if best_path:
        output_data = {
            "path": best_path,
            "start": start_point,
            "end": end_point,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"✅ Path saved to {output_file}")
    else:
        print("❌ No path generated due to emergency stop or invalid configuration.")

    return best_path
