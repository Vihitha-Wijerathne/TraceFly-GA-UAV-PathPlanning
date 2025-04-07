import numpy as np
from .GA_complex_env import ComplexEnvironment

class SimulationRunner:
    def __init__(self, x_bounds, y_bounds, z_bounds, num_obstacles):
        self.environment = ComplexEnvironment(x_bounds, y_bounds, z_bounds, num_obstacles)

    def simulate_path(self, start, destination):
        """
        Simulates a path from the start to the destination while avoiding obstacles.
        """
        current_position = np.array(start)
        destination = np.array(destination)
        path = [tuple(current_position)]
        visited_positions = set()

        while not np.array_equal(current_position, destination):
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

    def plot_simulation(self, path):
        """
        Plots the environment and the simulated path.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the environment
        self.environment.plot_environment(ax)

        # Plot the path
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color="blue", marker="o", label="Path")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()