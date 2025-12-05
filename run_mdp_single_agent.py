import argparse
import numpy as np
import matplotlib.pyplot as plt

import config
from policy.rollout import RolloutPlanner, RandomPolicy, GreedyPolicy, RandomGreedyPolicy, AStarPolicy
from mdp.simulator import Simulator
from mdp.visualization import setup_live_plot, update_live_plot, visualize_path
from mdp.drone import DroneMDP
from drone.state import DroneState
from drone.action import DroneAction

seed = config.seed
mdp = None

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
    global mdp
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
        base_policy=AStarPolicy,
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

    while not simulator.is_done():
        action, value = planner.select_action(simulator.state, simulator.tick)
        print(f"planner.select_action - Tick: {simulator.tick}, State: {simulator.state}, Action: {action}, Expected Value: {value}")
        
        next_state, reward, done = simulator.step(action)
        print(f"simulator.step - Tick: {simulator.tick}, Next State: {next_state}, Reward: {reward}, Done: {done}")

        print()
        
        # Update visualization after each step
        if config.LIVE_UPDATE:
            update_live_plot(fig, ax, simulator, reward, seed, mdp)

    if config.LIVE_UPDATE:
        plt.ioff()  # Turn off interactive mode
    visualize_path(simulator, seed, mdp)  # Show final plot
    
    print(f"Total Reward: {simulator.get_total_reward()}")

    # Print summary
    # print("\n" + "=" * 60)
    # print("SIMULATION SUMMARY")
    # print("=" * 60)
    # print(f"Test Case: {result['test_case']}")
    # print(f"Result: {'SUCCESS ✓' if result['success'] else 'FAILED ✗'}")
    # print(f"Steps: {result['steps']}")
    # print(f"Final Position: {result['final_position']}")
    # print("=" * 60)

if __name__ == "__main__":
    configure()
    main()