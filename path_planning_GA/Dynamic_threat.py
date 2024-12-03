import numpy as np
import matplotlib.pyplot as plt

class DynamicThreatAssessment:
    def __init__(self, uav_max_speed=5, uav_reaction_time=0.5, safety_distance=5):
        """
        Initializes the dynamic threat assessment system.
        
        Parameters:
        - uav_max_speed: Maximum speed of the UAV (m/s).
        - uav_reaction_time: UAV's reaction time to threats (seconds).
        - safety_distance: Minimum safe distance to maintain from threats (meters).
        """
        self.uav_max_speed = uav_max_speed
        self.uav_reaction_time = uav_reaction_time
        self.safety_distance = safety_distance

    def predict_collision(self, uav_position, uav_velocity, threat_position, threat_velocity):
        """
        Predicts if a collision will occur and the time to collision.
        
        Parameters:
        - uav_position: UAV's current position [x, y, z].
        - uav_velocity: UAV's velocity vector [vx, vy, vz].
        - threat_position: Threat's current position [x, y, z].
        - threat_velocity: Threat's velocity vector [vx, vy, vz].
        
        Returns:
        - is_collision: True if a collision is predicted, False otherwise.
        - time_to_collision: Estimated time to collision (seconds).
        """
        relative_position = np.array(threat_position) - np.array(uav_position)
        relative_velocity = np.array(threat_velocity) - np.array(uav_velocity)

        # Quadratic equation to solve for time to collision
        a = np.dot(relative_velocity, relative_velocity)
        b = 2 * np.dot(relative_position, relative_velocity)
        c = np.dot(relative_position, relative_position) - self.safety_distance**2

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return False, None  # No collision predicted

        # Calculate the smallest positive root (time to collision)
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        time_to_collision = min(t for t in [t1, t2] if t > 0)

        return time_to_collision > 0, time_to_collision

    def evade_threat(self, uav_position, threat_position, static_obstacles):
        """
        Computes an evasive maneuver to avoid the threat.
        
        Parameters:
        - uav_position: UAV's current position [x, y, z].
        - threat_position: Threat's current position [x, y, z].
        - static_obstacles: List of static obstacle positions [[x, y, z], ...].
        
        Returns:
        - evasion_position: New UAV position after evasion maneuver.
        - action_taken: Description of the evasion action.
        """
        # Possible moves: [front, back, left, right, up, down]
        moves = [
            (1, 0, 0), (-1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, 1), (0, 0, -1)
        ]

        # Evaluate each move for safety
        safe_moves = []
        for move in moves:
            new_position = [uav_position[i] + move[i] for i in range(3)]
            is_safe = all(
                np.linalg.norm(np.array(new_position) - np.array(obs)) > self.safety_distance
                for obs in static_obstacles + [threat_position]
            )
            if is_safe:
                safe_moves.append((move, new_position))

        if not safe_moves:
            return uav_position, "Emergency stop! No safe moves available."

        # Choose the move that maximizes distance from the threat
        chosen_move, evasion_position = max(
            safe_moves,
            key=lambda x: np.linalg.norm(np.array(x[1]) - np.array(threat_position))
        )
        direction = ['front', 'back', 'left', 'right', 'up', 'down'][moves.index(chosen_move)]

        return evasion_position, f"Moved {direction} to evade threat."

    def process_data(self, uav_position, uav_velocity, threat_position, threat_velocity, static_obstacles):
        """
        Processes real-time data and decides on threat evasion.
        
        Parameters:
        - uav_position: UAV's current position [x, y, z].
        - uav_velocity: UAV's velocity vector [vx, vy, vz].
        - threat_position: Threat's current position [x, y, z].
        - threat_velocity: Threat's velocity vector [vx, vy, vz].
        - static_obstacles: List of static obstacle positions [[x, y, z], ...].
        
        Returns:
        - action_taken: Description of the action taken.
        - evasion_position: New UAV position after evasion.
        """
        # Predict collision
        is_collision, time_to_collision = self.predict_collision(
            uav_position, uav_velocity, threat_position, threat_velocity
        )
        if not is_collision or time_to_collision > self.uav_reaction_time:
            return "No immediate threat detected.", uav_position

        # Evade the threat
        return self.evade_threat(uav_position, threat_position, static_obstacles)


# Example Usage
if __name__ == "__main__":
    # Initialize the threat assessment module
    threat_assessment = DynamicThreatAssessment()

    # Example UAV and threat data
    uav_position = [10, 10, 5]
    uav_velocity = [1, 0, 0]
    threat_position = [12, 10, 5]
    threat_velocity = [-1, 0, 0]
    static_obstacles = [[9, 11, 5], [10, 12, 5], [11, 9, 5]]

    # Process the data
    action, new_position = threat_assessment.process_data(
        uav_position, uav_velocity, threat_position, threat_velocity, static_obstacles
    )

    # Output results
    print("Action Taken:", action)
    print("New UAV Position:", new_position)
