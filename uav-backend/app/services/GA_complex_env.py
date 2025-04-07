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
                    print(f"Point {point} is inside a cube obstacle at {position} with size {size}")
                    return True
            elif shape == "sphere":
                if np.linalg.norm(np.array(point) - np.array(position)) <= size:
                    print(f"Point {point} is inside a sphere obstacle at {position} with size {size}")
                    return True
            elif shape == "cylinder":
                if (abs(point[0] - position[0]) <= size and
                        abs(point[1] - position[1]) <= size and
                        abs(point[2] - position[2]) <= 2 * size):
                    print(f"Point {point} is inside a cylinder obstacle at {position} with size {size}")
                    return True
        return False

    def plot_environment(self, ax, path=None):
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
                # Ensure the argument to sqrt is non-negative
                valid_mask = size**2 - X**2 >= 0
                Y = np.zeros_like(X)
                Y[valid_mask] = np.sqrt(size**2 - X[valid_mask]**2) + position[1]
                ax.plot_surface(X, Y, Z, color="gray", alpha=0.5)

        # Plot the UAV path if provided
        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color="blue", marker="o", label="UAV Path")
            ax.legend()

def generate_uav_path(environment, start, destination, max_iterations=1000):
    """
    Generates a simple UAV path from start to destination while avoiding obstacles.
    """
    current_position = np.array(start)
    destination = np.array(destination)
    path = [tuple(current_position)]
    iterations = 0

    while not np.array_equal(current_position, destination):
        # Break if the maximum number of iterations is reached
        if iterations >= max_iterations:
            print("Maximum iterations reached. Path generation stopped.")
            break

        # Calculate the direction vector
        direction = destination - current_position
        step = direction / np.linalg.norm(direction)  # Normalize the direction vector
        next_position = current_position + step

        # Check for obstacles
        if environment.is_obstacle(next_position):
            # If there's an obstacle, adjust the path to avoid it
            next_position = current_position + np.random.uniform(-1, 1, size=3)

        # Update the current position
        current_position = np.round(next_position).astype(int)
        path.append(tuple(current_position))

        iterations += 1

    return path

# Create a complex environment
env = ComplexEnvironment((0, 20), (0, 20), (0, 10), num_obstacles=10)

# Generate a UAV path
start = (0, 0, 0)
destination = (15, 15, 8)
uav_path = generate_uav_path(env, start, destination)

# Plot the environment and the UAV path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
env.plot_environment(ax, path=uav_path)
plt.show()