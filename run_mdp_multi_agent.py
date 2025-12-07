import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

import config
from policy.multi_agent_rollout import MultiAgentRolloutPlanner, AStarPolicy
from mdp.simulator_multi_agent import MultiAgentSimulator
from mdp.drone_multi_agent import MultiAgentDroneMDP
from mdp.visualization_multi_agent import setup_live_plot_multi_agent, update_live_plot_multi_agent, visualize_path_multi_agent
from drone.state import DroneState
from drone.state_multi_agent import MultiAgentDroneState
from drone.action_multi_agent import MultiAgentAction

seed = config.seed

def configure():
    parser = argparse.ArgumentParser(description="Multi-Agent MDP Drone Navigation Simulation")
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

    # Quiet mode is not support for multi agent MDP
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
        config.NUM_AGENTS = len(test_case.multi_agent_start_pos)
        config.MULTI_AGENT_STARTING_POSITIONS = np.array(test_case.multi_agent_start_pos)
        config.MULTI_AGENT_GOAL_POSITIONS = np.array(test_case.multi_agent_goal_pos)
        config.OBSTACLES = np.array(test_case.obstacles)
        config.MAX_OBSTACLES = config.OBSTACLES.shape[0]
        config.AGENT_REWARDS = [config.GOAL_REWARD] * config.NUM_AGENTS
    except ValueError as e:
        print(f"Error: {e}")
        print("\nUse --list to see available test cases")
    

def main():
    mdp = MultiAgentDroneMDP()
    base_policy = AStarPolicy()
    
    agent_states = []
    for agent_id in range(config.NUM_AGENTS):
        agent_state = DroneState(
            position=config.MULTI_AGENT_STARTING_POSITIONS[agent_id], 
            velocity=np.array([0, 0]), 
            battery=config.BATTERY_MAX, 
            wind=config.WIND[0]
        )
        agent_states.append(agent_state)
    initial_joint_state = MultiAgentDroneState(agent_states)
    
    simulator = MultiAgentSimulator(
        multi_agent_mdp=mdp, 
        initial_joint_states=initial_joint_state, 
        seed=seed
    )
    
    planner = MultiAgentRolloutPlanner(
        mdp=mdp, 
        base_policy=base_policy,
        num_rollouts=config.ROLLOUT_NUM_ROLLOUTS, 
        max_depth=config.ROLLOUT_MAX_DEPTH, 
        seed=seed
    )

    # Set up live visualization
    if config.LIVE_UPDATE:
        plt.ion()  # Turn on interactive mode
        fig, ax = setup_live_plot_multi_agent(seed, mdp, base_policy)
    else:
        fig, ax = None, None

    # Start timing
    start_time = time.time()

    while not simulator.is_done():
        joint_action, values = planner.select_action(simulator.joint_states, simulator.tick)
        print(f"planner.select_action - Tick: {simulator.tick}")
        print(f"  Joint State: {simulator.joint_states}")
        print(f"  Joint Action: {joint_action}")
        print(f"  Expected Values: {values}")
        
        next_joint_states, rewards, done = simulator.step(joint_action)
        print(f"simulator.step - Tick: {simulator.tick}")
        print(f"  Next Joint State: {next_joint_states}")
        print(f"  Rewards: {rewards}")
        print(f"  Done: {done}")

        print()
        
        # Update visualization after each step
        if config.LIVE_UPDATE:
            elapsed_time = time.time() - start_time
            update_live_plot_multi_agent(fig, ax, simulator, rewards, seed, mdp, elapsed_time, base_policy)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    total_rewards = simulator.get_total_reward()
    print(f"Total Rewards per Agent: {total_rewards}")
    print(f"Sum of All Rewards: {np.sum(total_rewards):.2f}")
    print(f"Simulation Time: {elapsed_time:.2f} seconds")

    if config.LIVE_UPDATE:
        plt.ioff()  # Turn off interactive mode
    visualize_path_multi_agent(simulator, seed, mdp, elapsed_time, base_policy)  # Show final plot

if __name__ == "__main__":
    configure()
    main()

