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
![image](https://github.com/user-attachments/assets/90e772d4-18ea-497e-97d0-d95898ef2013)


## Main Models and Contributors

### 1. **Genetic Algorithm-Based Path Planning**
**Student:** Wijerathne V.R  
**Overview:**  
Implements Genetic Algorithms (GAs) for real-time optimization of UAV paths. The module continuously evolves path solutions by simulating processes like selection, crossover, and mutation. The goal is to find the most efficient and safest path in dynamic environments.

Key Features:
- Chromosome representation for UAV paths.
- Fitness evaluation for safety, distance, and time.
- Integration with dynamic data from UAV sensors.

### **Algorithm Architecture**
![GA Component (3)](https://github.com/user-attachments/assets/2a978d52-8470-40c5-b6e1-458854aeea28)
---

### 2. **Ray-Casting-Based Obstacle Detection**
**Student:** Theekshana W.G.P  
**Overview:**  
Develops a Ray-Casting module integrated with LiDAR technology to enable precise obstacle detection and avoidance. This module enhances real-time mapping of hazards, ensuring safer navigation in cluttered environments.

Key Features:
- Utilizes LiDAR sensors for high-resolution mapping.
- Accurate real-time 3D mapping of obstacles.
- Integration with UAV control systems for adaptive navigation.
![image](https://github.com/user-attachments/assets/2493240a-ab12-4080-95c5-00eccc2efde3)

---

### 3. **Predictive Environmental Change Detection**
**Student:** Prabhanga K.G.B  
**Overview:**  
Focuses on dynamic spatiotemporal hazard prediction for Unmanned Aerial Vehicles (UAVs). It uses machine learning techniques like LSTM and CNN to predict environmental changes (e.g., wind speed, precipitation) and create real-time heat maps of potential hazards across different zones. This data helps UAVs make informed decisions for safer navigation.

Key Features:
- Combines time-based forecasting (LSTM) with spatial mapping (CNN) for real-time hazard heatmaps.
- Both historical weather data and simulated sensor data are used to improve prediction accuracy.
- Focuses on critical UAV factors like wind speed, precipitation, and visibility for safer navigation.
- Generates heatmaps and quantitative predictions to support UAV decision-making in dynamic environments.
![image](https://github.com/user-attachments/assets/baa3ec9b-d53b-4258-98e4-08a412b54109)

---

### 4. **Simulation Environment for Testing**
**Student:** De Silva K.P.C  
**Overview:**  
Design and develop a realistic simulation environment using MATLAB and Gazebo to validate UAV path-planning algorithms. The environment replicates disaster scenarios with dynamic elements such as debris, weather changes, and terrain shifts.

Key Features:
- Advanced physics-based simulations for UAV testing.
- Real-time environmental interactions, including moving obstacles.
- Iterative refinement of path-planning algorithms through simulation data.
![image](https://github.com/user-attachments/assets/954218c0-92ec-48a4-94a1-9bf88feb1980)

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


## To run the Application

1. in the frontend - `npm run dev`
2. in the backend - `uvicorn app.main:app --reload --port 8000`