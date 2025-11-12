import random
import numpy as np
import matplotlib.pyplot as plt

import config
from policy.rollout import RolloutPlanner, RandomPolicy, GreedyPolicy, RandomGreedyPolicy
from mdp.simulator import Simulator
from mdp.visualization import setup_live_plot, update_live_plot, visualize_path
from mdp.drone import DroneMDP
from drone.state import DroneState
from drone.action import DroneAction

seed = random.randint(0, 1000)

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
        base_policy=RandomPolicy,  # Pure random for better exploration
        num_rollouts=config.ROLLOUT_NUM_ROLLOUTS, 
        max_depth=config.ROLLOUT_MAX_DEPTH, 
        seed=seed
    )

    # Set up live visualization
    if config.LIVE_UPDATE:
        plt.ion()  # Turn on interactive mode
        fig, ax = setup_live_plot(seed)
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
            update_live_plot(fig, ax, simulator, reward, seed)

    if config.LIVE_UPDATE:
        plt.ioff()  # Turn off interactive mode
    visualize_path(simulator, seed)  # Show final plot
    
    print(f"Total Reward: {simulator.get_total_reward()}")

if __name__ == "__main__":
    main()