import argparse
import pomdp_py
import numpy as np
import time

import config
from pomdp.agent import *
from pomdp.state_observation import *
from pomdp.visualization import GridVisualizer

seed = config.seed

def run_simulation(test_case: config.TestCase, max_steps: int = 500, verbose: bool = True):
    """
    Run the POMDP drone simulation with a given test case.
    
    Args:
        test_case: The test case configuration to run
        max_steps: Maximum number of simulation steps
        verbose: Whether to print step-by-step output
    
    Returns:
        dict with simulation results (success, steps, final_position)
    """
    # Extract test case configuration
    start_pos = Coordinate(*test_case.start_pos)
    goal_pos = Coordinate(*test_case.goal_pos)
    true_obstacles = [Coordinate(x, y) for x, y in test_case.obstacles]
    
    if verbose:
        print(f"{'='*60}")
        print(f"Test Case: {test_case.name}")
        print(f"Difficulty: {test_case.difficulty}")
        print(f"Description: {test_case.description}")
        print(f"Start: {start_pos}, Goal: {goal_pos}")
        print(f"Obstacles: {len(true_obstacles)} cells")
        print(f"{'='*60}\n")
    
    # Init Agent
    agent = RobotAgent(start_pos, goal_pos, DIRECTIONS)
    
    # Init Planner (POMCP) - use shared pathfinder from agent
    rollout_policy = GridRolloutPolicy(ACTIONS, agent.pathfinder)
    planner = pomdp_py.POMCP(
        max_depth=15, 
        exploration_const=0.0, 
        num_sims=500, 
        rollout_policy=rollout_policy
    )

    if verbose:
        print("Starting Simulation...")
    
    # Mock Real World (The Truth)
    true_robot_pos = start_pos
    
    # Initialize visualizer
    viz = GridVisualizer(
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        obstacles=true_obstacles,
        goal=goal_pos,
        start_pos=start_pos,
        agent=agent  # Pass agent to access occupancy grid for belief visualization
    )
    
    # Initialize reward tracking
    total_reward = 0.0
    
    # Start timing
    start_time = time.time()
    viz.update(true_robot_pos, step=0, action_name="start", reward=0.0, total_reward=0.0, elapsed_time=0.0)
    
    # Simulation loop
    success = False
    final_step = max_steps
    
    for step in range(max_steps):
        # 1. Plan
        action = planner.plan(agent)
        if verbose:
            print(f"Step {step}: Moving {action.name}")
        
        # 2. Store previous state for reward calculation
        prev_state = GridState(true_robot_pos, true_obstacles, goal_pos)
        
        # 3. Execute (Update True World)
        nx = true_robot_pos.x + action.delta[0]
        ny = true_robot_pos.y + action.delta[1]
        new_pos = Coordinate(nx, ny)
        
        if new_pos in true_obstacles or not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
            if verbose:
                print("  -> Bonk! Hitting wall.")
        else:
            true_robot_pos = new_pos
            
        if verbose:
            print(f"  -> True Pos: {true_robot_pos}")
        
        # 4. Calculate reward for this transition
        next_state = GridState(true_robot_pos, true_obstacles, goal_pos)
        step_reward = agent.reward_model.sample(prev_state, action, next_state)
        total_reward += step_reward
        
        if verbose:
            print(f"  -> Reward: {step_reward:.2f}, Total: {total_reward:.2f}")
        
        # 5. Update visualization with rewards
        elapsed_time = time.time() - start_time
        viz.update(true_robot_pos, step=step+1, action_name=action.name, 
                  reward=step_reward, total_reward=total_reward, elapsed_time=elapsed_time)
        
        # 6. Observe (Generate observation based on sensor accuracy)
        real_obs = agent.observation_model.sample(next_state, action)
        
        # 7. Manual belief update
        agent.manual_belief_update(action, real_obs, true_robot_pos)
        
        # 8. Reset tree so POMCP starts fresh with the new belief
        agent.tree = None
        
        # Check Goal
        if true_robot_pos == goal_pos:
            if verbose:
                print(f"\nGoal Reached in {step + 1} steps!")
            viz.show_goal_reached()
            success = True
            final_step = step + 1
            break
    
    if not success and verbose:
        print(f"\nFailed to reach goal within {max_steps} steps")
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Keep window open at the end
    viz.show()
    
    return {
        "success": success,
        "steps": final_step,
        "final_position": true_robot_pos,
        "total_reward": total_reward,
        "elapsed_time": elapsed_time,
        "test_case": test_case.name
    }


def main():
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
    # Wind, velocity, and acceleration are not supported for POMDP
    # parser.add_argument(
    #     "--wind", "-w",
    #     action="store_true",
    #     default=config.WIND_ENABLE,
    #     help=f"Enable wind effects during simulation (default: {config.WIND_ENABLE})"
    # )
    # parser.add_argument(
    #     "--max-velocity", "-v",
    #     type=int,
    #     default=config.VELOCITY_MAX,
    #     help=f"Maximum velocity (default: {config.VELOCITY_MAX})"
    # )
    # parser.add_argument(
    #     "--max-acceleration", "-a",
    #     type=int,
    #     default=config.ACCEL_MAX,
    #     help=f"Maximum acceleration (default: {config.ACCEL_MAX})"
    # )
    parser.add_argument(
        "--max-steps", "-m",
        type=int,
        default=500,
        help="Maximum number of simulation steps"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress step-by-step output"
    )
    
    args = parser.parse_args()
    
    # Apply configuration overrides
    if args.seed is not None:
        global seed
        seed = args.seed

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
        return
    
    # Run simulation
    try:
        test_case = config.get_test_case(args.test_case)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nUse --list to see available test cases")
        return
    
    result = run_simulation(
        test_case=test_case,
        max_steps=args.max_steps,
        verbose=not args.quiet
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Test Case: {result['test_case']}")
    print(f"Steps: {result['steps']}")
    print(f"Final Position: {result['final_position']}")
    print(f"Total Reward: {result['total_reward']:.2f}")
    print(f"Elapsed Time: {result['elapsed_time']:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
