import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CollisionAvoidanceModule:
    def __init__(self, safety_distance=5, max_safe_speed=5):
        """
        Initializes the collision avoidance module.
        
        Parameters:
        - safety_distance: Minimum safe distance from obstacles (meters).
        - max_safe_speed: Maximum UAV speed considered safe near obstacles (m/s).
        """
        self.safety_distance = safety_distance
        self.max_safe_speed = max_safe_speed

    def find_closest_obstacle(self, lidar_data, uav_position, obstacle_positions):
        """
        Finds the closest obstacle that is most likely to collide with the UAV.
        Parameters:
        - lidar_data: Array of distances from obstacles in different directions.
        - uav_position: Current UAV position in 3D space.
        - obstacle_positions: List of all obstacle positions (x, y, z).

        Returns:
        - Closest obstacle position.
        - Index of the closest obstacle.
        """
        distances = [
            np.linalg.norm(np.array(uav_position) - np.array(obs))
            for obs in obstacle_positions
        ]
        closest_idx = np.argmin(distances)
        return obstacle_positions[closest_idx], closest_idx

    def avoid_collision(self, closest_obstacle, obstacle_positions, uav_position):
        """
        Takes immediate action to avoid the closest obstacle without hitting others.

        Parameters:
        - closest_obstacle: The obstacle to avoid.
        - obstacle_positions: All obstacle positions.
        - uav_position: Current UAV position.

        Returns:
        - Updated UAV position after collision avoidance.
        - Action taken (string describing the movement).
        """
        directions = ['front', 'back', 'left', 'right', 'up', 'down']
        moves = [
            (1, 0, 0), (-1, 0, 0),  # Front, Back
            (0, -1, 0), (0, 1, 0),  # Left, Right
            (0, 0, 1), (0, 0, -1)   # Up, Down
        ]

        # Check safe moves
        safe_moves = []
        for move in moves:
            new_position = [uav_position[0] + move[0], uav_position[1] + move[1], uav_position[2] + move[2]]
            if all(
                np.linalg.norm(np.array(new_position) - np.array(obs)) > self.safety_distance
                for obs in obstacle_positions
            ):
                safe_moves.append((move, new_position))

        if not safe_moves:
            return uav_position, "No safe moves available! Emergency stop."

        # Choose the safest move based on distance from the closest obstacle
        chosen_move, updated_position = min(
            safe_moves,
            key=lambda x: np.linalg.norm(np.array(x[1]) - np.array(closest_obstacle))
        )
        chosen_direction = directions[moves.index(chosen_move)]

        return updated_position, f"Moved {chosen_direction} to avoid collision."

    def plot_avoidance(self, uav_position, closest_obstacle, obstacle_positions, updated_position, action_taken):
        """
        Plots the UAV's current position, the closest obstacle, all obstacles, and the chosen movement.

        Parameters:
        - uav_position: Current UAV position (x, y, z).
        - closest_obstacle: The obstacle being avoided.
        - obstacle_positions: All obstacle positions.
        - updated_position: New UAV position after collision avoidance.
        - action_taken: Description of the movement action.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot all obstacles
        obstacle_positions = np.array(obstacle_positions)
        ax.scatter(obstacle_positions[:, 0], obstacle_positions[:, 1], obstacle_positions[:, 2], c='red', label='Obstacles')

        # Highlight the closest obstacle
        ax.scatter(*closest_obstacle, color='orange', s=150, label='Closest Obstacle')

        # Plot UAV's current position
        ax.scatter(*uav_position, color='green', s=150, label='UAV Current Position')

        # Plot updated position
        ax.scatter(*updated_position, color='blue', s=150, label='Updated Position')
        ax.quiver(
            uav_position[0], uav_position[1], uav_position[2],
            updated_position[0] - uav_position[0],
            updated_position[1] - uav_position[1],
            updated_position[2] - uav_position[2],
            color='blue', label='Movement Direction'
        )

        # Add action text
        ax.text(
            updated_position[0], updated_position[1], updated_position[2],
            f"Action: {action_taken}",
            color='black', fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.legend()
        plt.show()


# Main Function to Test Collision Avoidance
if __name__ == "__main__":
    # Initialize the module with default safety parameters
    collision_avoidance = CollisionAvoidanceModule(safety_distance=5, max_safe_speed=5)

    # Example input data
    uav_position = [10, 10, 5]  # Current UAV position
    obstacle_positions = [
        [11, 10, 5], [8, 12, 5], [10, 9, 6], [10, 15, 5], [12, 10, 6]
    ]  # Obstacle positions

    # Simulate LiDAR distances
    lidar_data = [
        np.linalg.norm(np.array(uav_position) - np.array(obs))
        for obs in obstacle_positions
    ]

    # Find the closest obstacle
    closest_obstacle, _ = collision_avoidance.find_closest_obstacle(lidar_data, uav_position, obstacle_positions)

    # Avoid the closest obstacle
    updated_position, action = collision_avoidance.avoid_collision(closest_obstacle, obstacle_positions, uav_position)

    # Display results
    print("Action Taken:", action)
    print("Updated Position:", updated_position)

    # Plot results
    collision_avoidance.plot_avoidance(uav_position, closest_obstacle, obstacle_positions, updated_position, action)
