import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from random import choice

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

# Create a complex environment
env = ComplexEnvironment((0, 20), (0, 20), (0, 10), num_obstacles=10)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
env.plot_environment(ax)
plt.show()
