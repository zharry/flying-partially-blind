import random
import numpy as np
import matplotlib.pyplot as plt

import config
from simulator import Simulator
from rollout import RolloutPlanner, RandomPolicy
from drone.drone_mdp import DroneMDP
from drone.state import DroneState
from drone.action import DroneAction

seed = 228 #random.randint(0, 1000)

def main():
    mdp = DroneMDP()
    initial_state = DroneState(
        position=config.STARTING_POSITION[0], 
        velocity=np.array([0, 0]), 
        battery=config.BATTERY_MAX, 
        wind=config.WIND[0]
    )
    simulator = Simulator(
        mdp=mdp, 
        initial_state=initial_state, 
        seed=seed
    )
    planner = RolloutPlanner(
        mdp=mdp, 
        base_policy=RandomPolicy, 
        num_rollouts=10, 
        max_depth=5, 
        seed=seed
    )

    while not simulator.is_done():
        action, value = planner.select_action(simulator.state, simulator.tick)
        print(f"planner.select_action - Tick: {simulator.tick}, State: {simulator.state}, Action: {action}, ExpectedValue: {value}")
        
        next_state, reward, done = simulator.step(action)
        print(f"simulator.step - Tick: {simulator.tick}, Next State: {next_state}, Reward: {reward}, Done: {done}")

        print()

    visualize_path(simulator)
    
    print(f"Total Reward: {simulator.get_total_reward()}")

 

def visualize_path(simulator: Simulator):
    plt.figure(figsize=(config.GRID_SIZE, config.GRID_SIZE))
    drone_positions = np.array([state.position for state in simulator.state_history])
    
    # Plot obstacles as red circles with size corresponding to OBSTACLE_THRESHOLD
    if len(config.OBSTACLES) > 0:
        # Marker area size = (radius in points^2)*pi
        # Convert OBSTACLE_THRESHOLD (in grid units) to marker size in points^2 for scatter:
        # Each grid cell is 1x1, and matplotlib's scatter s parameter is marker area in points^2.
        # Assuming a figure of size (GRID_SIZE, GRID_SIZE) and default dpi, roughly estimate:
        # We'll use a heuristic: make s = (OBSTACLE_THRESHOLD * 72)^2
        # 1 grid unit ≈ 72 points (1 inch in default dpi), scale by threshold
        # Estimate of 1 grid unit in points for a 2000x2000 pixel figure (presumably 2000x2000 points at 100 dpi)
        # There are GRID_SIZE (e.g. 40) grid cells across 2000 points, so:
        points_per_grid_unit = config.PLOT_FIGURE_SIZE / config.GRID_SIZE
        obstacle_marker_size = (config.OBSTACLE_THRESHOLD * points_per_grid_unit) ** 2
        plt.scatter(
            config.OBSTACLES[:, 0], config.OBSTACLES[:, 1],
            c='red', s=obstacle_marker_size, marker='o',
            label='Obstacles', zorder=3
        )
    
    # Plot goal as green star
    plt.scatter(config.GOAL_POSITION[0], config.GOAL_POSITION[1], 
               c='green', s=300, marker='*', label='Goal', zorder=4)
    
    # Plot starting position as blue circle
    plt.scatter(drone_positions[0, 0], drone_positions[0, 1], 
               c='blue', s=150, marker='o', label='Start', zorder=5)
    
    # Plot drone path as a line with markers
    plt.plot(drone_positions[:, 0], drone_positions[:, 1], 
            'b-', linewidth=2, alpha=0.6, label='Drone Path', zorder=2)
    plt.scatter(drone_positions[:, 0], drone_positions[:, 1], 
               c='cyan', s=50, marker='o', alpha=0.7, zorder=2)
    
    # Set plot limits and labels
    plt.xlim(-0.5, config.GRID_SIZE - 0.5)
    plt.ylim(-0.5, config.GRID_SIZE - 0.5)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Drone Path Visualization (Seed: {seed})\nTotal Reward: {simulator.get_total_reward():.2f}\nTotal Steps: {len(simulator.state_history)}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    fig = plt.gcf()
    fig.set_size_inches(config.PLOT_FIGURE_SIZE / fig.dpi, config.PLOT_FIGURE_SIZE / fig.dpi)
    
    plt.show()

if __name__ == "__main__":
    main()