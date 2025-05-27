import time
import threading
import json
import os
from path_planning_GA.path_planner_hrmaga import HRMAGAGenerator
from obstacle_detection.Hybrid_Model_Validation.lidar_hybrid_model_validate import LidarModule

# Initialize Lidar Module and HR-MAGA GA
lidar = LidarModule()
ga_system = HRMAGAGenerator()

# Emergency flag
emergency_stop = threading.Event()

def monitor_emergency():
    """Continuously monitor for an emergency stop command."""
    while not emergency_stop.is_set():
        try:
            with open("emergency_flag.json", "r") as f:
                status = json.load(f)
                if status.get("emergency_stop", False):
                    emergency_stop.set()
                    ga_system.emergency_halt()
                    print("\n🚨 Emergency Stop Triggered! UAV halted.")
        except FileNotFoundError:
            pass
        time.sleep(1)

def update_lidar():
    """Continuously simulate LiDAR scanning and update detected objects."""
    while not emergency_stop.is_set():
        lidar.scan_environment()
        time.sleep(3)  # Frequency of obstacle updates

def run_ga_path_planning():
    """Run path planning using GA from segment to segment."""
    current_pos = (0, 0, 0)
    destination = (30, 30, 8)
    segment_length = 5

    while not emergency_stop.is_set():
        # Compute direction vector
        direction = (
            destination[0] - current_pos[0],
            destination[1] - current_pos[1],
            destination[2] - current_pos[2]
        )
        norm = max((sum([i ** 2 for i in direction]) ** 0.5), 0.001)
        unit_direction = tuple(i / norm for i in direction)

        # Determine next local waypoint
        next_waypoint = (
            current_pos[0] + unit_direction[0] * segment_length,
            current_pos[1] + unit_direction[1] * segment_length,
            current_pos[2] + unit_direction[2] * segment_length,
        )

        # Call GA module to generate local path
        best_path = ga_system.generate_local_path(current_pos, next_waypoint)

        # If path is empty or stopped, terminate
        if not best_path or emergency_stop.is_set():
            print("\n🛑 Path generation halted or failed.")
            break

        # Move to last waypoint of segment
        current_pos = best_path[-1]

        print(f"\n✅ Moved to {current_pos}")

        # Check if destination reached
        if all(abs(current_pos[i] - destination[i]) < 2 for i in range(3)):
            print("\n🎯 Destination reached!")
            break

        time.sleep(1)  # Simulate UAV travel delay

if __name__ == "__main__":
    print("\n🚀 Starting UAV Path Planning System")

    # Launch threads
    threads = [
        threading.Thread(target=monitor_emergency),
        threading.Thread(target=update_lidar),
        threading.Thread(target=run_ga_path_planning)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print("\n✅ UAV Mission Terminated.")
