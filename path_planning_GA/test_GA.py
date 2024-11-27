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
        # Randomly generate obstacle coordinates within bounds
        obstacles = []
        for _ in range(self.num_obstacles):
            x = np.random.randint(*self.x_bounds)
            y = np.random.randint(*self.y_bounds)
            z = np.random.randint(*self.z_bounds)
            obstacles.append((x, y, z))
        return obstacles

# UAV Path Planner with Optimized Logic
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
        # Compute points along the straight line between start and end
        points = []
        num_steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), abs(end[2] - start[2]))
        for t in np.linspace(0, 1, num_steps + 1):
            x = int(round(start[0] + t * (end[0] - start[0])))
            y = int(round(start[1] + t * (end[1] - start[1])))
            z = int(round(start[2] + t * (end[2] - start[2])))
            points.append((x, y, z))
        return points

    def find_next_step(self, current, target):
        # Generate a direct step towards the target
        direction = (
            np.sign(target[0] - current[0]),
            np.sign(target[1] - current[1]),
            np.sign(target[2] - current[2]),
        )
        next_point = (current[0] + direction[0], current[1] + direction[1], current[2] + direction[2])
        return next_point

    def dodge_obstacle(self, current, target):
        # Find a feasible detour around the obstacle
        possible_moves = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
        for move in possible_moves:
            next_point = (current[0] + move[0], current[1] + move[1], current[2] + move[2])
            if self.is_within_bounds(next_point) and not self.is_collision(next_point):
                return next_point
        return current  # If no move is possible (edge case)

    def find_path(self):
        current = self.source
        while current != self.destination:
            straight_line = self.compute_straight_line(current, self.destination)
            for point in straight_line:
                if self.is_collision(point):
                    # Dodge obstacle
                    current = self.dodge_obstacle(current, self.destination)
                    break
                else:
                    self.path.append(point)
                    current = point
                    if current == self.destination:
                        break
        return self.path

    def plot_path(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot obstacles
        obstacles = np.array(self.environment.obstacles)
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], c='red', label='Obstacles')

        # Plot source and destination
        ax.scatter(*self.source, c='blue', s=100, label='Source')
        ax.scatter(*self.destination, c='purple', s=100, label='Destination')

        # Plot the path
        path = np.array(self.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c='green', linewidth=2, label='Path')

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
environment = Environment(x_bounds, y_bounds, z_bounds, num_obstacles=200)

# Run the UAV Path Planner
planner = UAVPathPlanner(source, destination, environment)
path = planner.find_path()
planner.plot_path()