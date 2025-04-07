from services.simulation_runner import SimulationRunner

if __name__ == "__main__":
    # Define the environment bounds and number of obstacles
    x_bounds = (0, 20)
    y_bounds = (0, 20)
    z_bounds = (0, 10)
    num_obstacles = 10

    # Initialize the simulation runner
    runner = SimulationRunner(x_bounds, y_bounds, z_bounds, num_obstacles)

    # Define the start and destination points
    start = (0, 0, 0)
    destination = (15, 15, 8)

    # Simulate the path
    path = runner.simulate_path(start, destination)
    print("Simulated Path:", path)

    # Plot the simulation
    runner.plot_simulation(path)