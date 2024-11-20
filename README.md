# TraceFly

## **Genetic Algorithms-Based Unmanned Aerial Vehicle (UAV) Path Planning in Dynamic Environments**

## Overview
This research focuses on the development of a robust path-planning system for Unmanned Aerial Vehicles (UAVs) using **Genetic Algorithms (GAs)**. The system is designed to navigate dynamic environments effectively, addressing challenges such as unpredictable weather, debris, and terrain variations commonly found in disaster scenarios. The project integrates real-time sensor data, predictive modeling, and advanced optimization techniques to ensure safe, efficient, and adaptive navigation.

The project aims to:
- Develop a simulation environment replicating real-world disaster scenarios.
- Optimize UAV paths using GAs combined with real-time sensor data.
- Enhance UAV navigation with predictive environmental models.
- Validate the system through simulations using MATLAB and Gazebo.

---

## Main Models and Contributors

### 1. **Genetic Algorithm-Based Path Planning**
**Student:** Wijerathne V.R  
**Overview:**  
Implements Genetic Algorithms (GAs) for real-time optimization of UAV paths. The module continuously evolves path solutions by simulating processes like selection, crossover, and mutation. The goal is to find the most efficient and safest path in dynamic environments.

Key Features:
- Chromosome representation for UAV paths.
- Fitness evaluation for safety, distance, and time.
- Integration with dynamic data from UAV sensors.

---

### 2. **Ray-Casting-Based Obstacle Detection**
**Student:** Theekshana W.G.P  
**Overview:**  
Develops a Ray-Casting module integrated with LiDAR technology to enable precise obstacle detection and avoidance. This module enhances real-time mapping of hazards, ensuring safer navigation in cluttered environments.

Key Features:
- Utilizes LiDAR sensors for high-resolution mapping.
- Accurate real-time 3D mapping of obstacles.
- Integration with UAV control systems for adaptive navigation.

---

### 3. **Predictive Environmental Change Detection**
**Student:** Prabhanga K.G.B  
**Overview:**  
Focuses on predictive modeling to forecast environmental changes, such as shifting weather and terrain. By leveraging machine learning models, this module enables UAVs to proactively adjust their paths in response to hazards.

Key Features:
- Real-time hazard prediction using LSTM and Random Forest.
- Integration of historical and sensor data for accurate forecasts.
- Provides data to the GA module for preemptive path adjustments.

---

### 4. **Simulation Environment for Testing**
**Student:** De Silva K.P.C  
**Overview:**  
Designs and develops a realistic simulation environment using tools like MATLAB and Gazebo to validate UAV path-planning algorithms. The environment replicates disaster scenarios with dynamic elements such as debris, weather changes, and terrain shifts.

Key Features:
- Advanced physics-based simulations for UAV testing.
- Real-time environmental interactions, including moving obstacles.
- Iterative refinement of path-planning algorithms through simulation data.

---

## System Workflow
1. **Data Input:** Collect real-time data from UAV sensors (e.g., LiDAR, weather sensors).
2. **Processing:**
   - Predict environmental changes.
   - Detect and map obstacles.
   - Optimize paths using Genetic Algorithms.
3. **Output:** Generate safe and efficient UAV paths, with continuous adaptation during flight.

---

## Authors
- **Wijerathne V.R**
- **Theekshana W.G.P**
- **Prabhanga K.G.B**
- **De Silva K.P.C**

### Supervisors
- **Dr. Sanika Wijayasekara**
- **Ms. Ishara Weerathunga**
