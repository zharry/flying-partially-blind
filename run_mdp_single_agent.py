import argparse
from tkinter import Grid
import numpy as np
import matplotlib.pyplot as plt
import time

import config
from policy.rollout import RolloutPlanner, RandomPolicy, GreedyPolicy, RandomGreedyPolicy, AStarPolicy
from mdp.simulator import Simulator
from mdp.visualization import setup_live_plot, update_live_plot, visualize_path
from mdp.drone import DroneMDP
from drone.state import DroneState
from drone.action import DroneAction

seed = config.seed

def configure():
    parser = argparse.ArgumentParser(description="POMDP Drone Navigation Simulation")
    parser.add_argument(
        "--test-case", "-t",
        type=str,
        default="easy_open_field",
        help="Name of the test case to run"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available test cases"
    )
    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        choices=["easy", "medium", "hard"],
        help="Filter test cases by difficulty (used with --list)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=config.seed,
        help=f"Random seed for simulation (default: random)"
    )
    parser.add_argument(
        "--wind", "-w",
        action="store_true",
        default=config.WIND_ENABLE,
        help=f"Enable wind effects during simulation (default: {config.WIND_ENABLE})"
    )
    parser.add_argument(
        "--max-velocity", "-v",
        type=int,
        default=config.VELOCITY_MAX,
        help=f"Maximum velocity (default: {config.VELOCITY_MAX})"
    )
    parser.add_argument(
        "--max-acceleration", "-a",
        type=int,
        default=config.ACCEL_MAX,
        help=f"Maximum acceleration (default: {config.ACCEL_MAX})"
    )
    # To configure max steps see config.py for simulation and MDP parameters
    # parser.add_argument(
    #     "--max-steps", "-m",
    #     type=int,
    #     default=500,
    #     help="Maximum number of simulation steps"
    # )
    # Quiet mode is not support for single agent MDP
    # parser.add_argument(
    #     "--quiet", "-q",
    #     action="store_true",
    #     help="Suppress step-by-step output"
    # )
    
    args = parser.parse_args()
    
    # Apply configuration overrides
    if args.seed is not None:
        global seed
        seed = args.seed
        config.seed = args.seed
        config.rng = np.random.RandomState(args.seed)
        # Regenerate wind with new seed
        config.WIND = np.array([[config.rng.randint(config.WIND_MIN, config.WIND_MAX + 1), 
                                config.rng.randint(config.WIND_MIN, config.WIND_MAX + 1)] 
                               for _ in range(config.MAXIMUM_TIME_STEPS)])
        # Regenerate obstacles with new seed
        config.OBSTACLES = config.rng.randint(0, config.GRID_SIZE, size=(40, 2))
        config.MAX_OBSTACLES = config.OBSTACLES.shape[0]
    
    if not args.wind:
        config.WIND = np.array([[0, 0] for _ in range(config.MAXIMUM_TIME_STEPS)])

    if args.max_velocity is not None:
        config.VELOCITY_MAX = args.max_velocity
        config.VELOCITY_MIN = -args.max_velocity
    
    if args.max_acceleration is not None:
        config.ACCEL_MAX = args.max_acceleration
        config.ACCEL_MIN = -args.max_acceleration

    print("Parsed Configuration:")
    print(f"  Seed: {config.seed}")
    print(f"  Wind Enabled: {config.WIND_ENABLE}")
    print(f"  Velocity: {config.VELOCITY_MAX}, {config.VELOCITY_MIN}")
    print(f"  Acceleration: {config.ACCEL_MAX}, {config.ACCEL_MIN}")
    print(f"  Max Steps: {config.MAXIMUM_TIME_STEPS}")
    print()

    # List mode
    if args.list:
        print("\nAvailable Test Cases:")
        print("-" * 60)
        for name in config.list_test_cases(args.difficulty):
            tc = config.TEST_CASES[name]
            print(f"  [{tc.difficulty:6}] {name:25} - {tc.description[:40]}...")
        print()
        exit()
    
    # Get configuration and convert to legacy format
    try:
        test_case = config.get_test_case(args.test_case)
        config.OBSTACLES = np.array(test_case.obstacles)
        config.MAX_OBSTACLES = config.OBSTACLES.shape[0]
        config.STARTING_POSITION =np.array([test_case.start_pos])
        config.GOAL_POSITION = np.array(test_case.goal_pos)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nUse --list to see available test cases")
    

def main():
    mdp = DroneMDP()
    base_policy = AStarPolicy()

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
        base_policy=base_policy,
        num_rollouts=config.ROLLOUT_NUM_ROLLOUTS, 
        max_depth=config.ROLLOUT_MAX_DEPTH, 
        seed=seed
    )

    # Set up live visualization
    if config.LIVE_UPDATE:
        plt.ion()  # Turn on interactive mode
        fig, ax = setup_live_plot(seed, mdp)
    else:
        fig, ax = None, None

    # Start timing
    start_time = time.time()

    while not simulator.is_done():
        action, value = planner.select_action(simulator.state, simulator.tick)
        print(f"planner.select_action - Tick: {simulator.tick}")
        print(f"  State: {simulator.state}")
        print(f"  Action: {action}")
        print(f"  Expected Value: {value}")
        
        next_state, reward, done = simulator.step(action)
        print(f"simulator.step - Tick: {simulator.tick}")
        print(f"  Next State: {next_state}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")

        print()
        
        # Update visualization after each step
        if config.LIVE_UPDATE:
            elapsed_time = time.time() - start_time
            update_live_plot(fig, ax, simulator, reward, seed, mdp, elapsed_time)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total Reward: {simulator.get_total_reward()}")
    print(f"Simulation Time: {elapsed_time:.2f} seconds")

    if config.LIVE_UPDATE:
        plt.ioff()  # Turn off interactive mode
    visualize_path(simulator, seed, mdp, elapsed_time)  # Show final plot

if __name__ == "__main__":
    configure()
    main()