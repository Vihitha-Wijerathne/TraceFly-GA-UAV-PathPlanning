import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from .GA_complex_env import ComplexEnvironment

class SimulationRunner:
    def __init__(self, x_bounds, y_bounds, z_bounds, num_obstacles):
        self.environment = ComplexEnvironment(x_bounds, y_bounds, z_bounds, num_obstacles)

    def simulate_path(self, start, destination, steps):
        """
        Simulates a path from the start to the destination while avoiding obstacles.
        """
        current_position = np.array(start)
        destination = np.array(destination)
        path = [tuple(current_position)]
        visited_positions = set()

        for _ in range(steps):
            # Check if the UAV is stuck in a loop
            if tuple(current_position) in visited_positions:
                print("UAV is stuck in a loop. Terminating simulation.")
                break
            visited_positions.add(tuple(current_position))

            # Calculate the direction vector
            direction = destination - current_position
            step = direction / np.linalg.norm(direction)  # Normalize the direction vector
            next_position = current_position + step

            # Check for obstacles
            if self.environment.is_obstacle(next_position):
                # If there's an obstacle, adjust the path to avoid it
                next_position = self.avoid_obstacle(current_position, step)

            # Update the current position
            current_position = np.round(next_position).astype(int)
            path.append(tuple(current_position))

            # Stop if the destination is reached
            if np.array_equal(current_position, destination):
                break

        return path

    def avoid_obstacle(self, current_position, step):
        """
        Adjusts the path to avoid obstacles by slightly altering the direction.
        """
        for angle in np.linspace(-np.pi / 2, np.pi / 2, 20):  # Wider range of angles
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            new_step = np.dot(rotation_matrix, step)
            new_position = current_position + new_step

            if not self.environment.is_obstacle(new_position):
                return new_position

        # If no valid path is found, backtrack slightly
        backtrack_step = -0.5 * step
        backtrack_position = current_position + backtrack_step
        if not self.environment.is_obstacle(backtrack_position):
            return backtrack_position

        # If still stuck, return the current position
        print("No valid path found. UAV is stuck.")
        return current_position

    def plot_simulation(self, path, interval=5000):
        """
        Plots the environment and animates the simulated path as a line.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the environment
        self.environment.plot_environment(ax)

        # Plot start and destination points
        ax.scatter(*path[0], color="green", s=100, label="Start")
        ax.scatter(*path[-1], color="red", s=100, label="Destination")

        # Initialize the line for the drone's path
        drone_path, = ax.plot([], [], [], color="blue", label="Drone Path")
        ax.legend()

        def update(frame):
            current_path = np.array(path[:frame + 1])
            drone_path.set_data(current_path[:, 0], current_path[:, 1])
            drone_path.set_3d_properties(current_path[:, 2])
            return drone_path,

        # Animate the drone's movement
        ani = FuncAnimation(fig, update, frames=len(path), interval=interval, blit=False)
        plt.show()

# Example usage
if __name__ == "__main__":
    x_bounds = (0, 20)
    y_bounds = (0, 20)
    z_bounds = (0, 10)
    num_obstacles = 10
    start = (0, 0, 0)
    destination = (15, 15, 0)
    steps = 12 

    runner = SimulationRunner(x_bounds, y_bounds, z_bounds, num_obstacles)
    path = runner.simulate_path(start, destination, steps)
    runner.plot_simulation(path, interval=5000)