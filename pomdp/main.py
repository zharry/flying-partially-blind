import argparse
from state_observation import *
from agent import *
from config import (
    DIRECTIONS, GRID_WIDTH, GRID_HEIGHT, 
    TestCase, TEST_CASES, get_test_case, list_test_cases
)
from visualization import GridVisualizer
import pomdp_py


def run_simulation(test_case: TestCase, max_steps: int = 500, verbose: bool = True):
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
        print(f"\n{'='*60}")
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
        exploration_const=5.0, 
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
        goal=goal_pos
    )
    viz.update(true_robot_pos, step=0, action_name="start")
    
    # Simulation loop
    success = False
    final_step = max_steps
    
    for step in range(max_steps):
        # 1. Plan
        action = planner.plan(agent)
        if verbose:
            print(f"Step {step}: Moving {action.name}")
        
        # 2. Execute (Update True World)
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
        
        # Update visualization
        viz.update(true_robot_pos, step=step+1, action_name=action.name)
        
        # 3. Observe (Generate observation based on sensor accuracy)
        true_state = GridState(true_robot_pos, true_obstacles, goal_pos)
        real_obs = agent.observation_model.sample(true_state, action)
        
        # 4. Manual belief update
        agent.manual_belief_update(action, real_obs, true_robot_pos)
        
        # 5. Reset tree so POMCP starts fresh with the new belief
        agent.tree = None
        
        # Check Goal
        if true_robot_pos == goal_pos:
            if verbose:
                print(f"\n🎉 Goal Reached in {step + 1} steps!")
            viz.show_goal_reached()
            success = True
            final_step = step + 1
            break
    
    if not success and verbose:
        print(f"\n❌ Failed to reach goal within {max_steps} steps")
    
    # Keep window open at the end
    viz.show()
    
    return {
        "success": success,
        "steps": final_step,
        "final_position": true_robot_pos,
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
    
    # List mode
    if args.list:
        print("\nAvailable Test Cases:")
        print("-" * 60)
        for name in list_test_cases(args.difficulty):
            tc = TEST_CASES[name]
            print(f"  [{tc.difficulty:6}] {name:25} - {tc.description[:40]}...")
        print()
        return
    
    # Run simulation
    try:
        test_case = get_test_case(args.test_case)
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
    print(f"Result: {'SUCCESS ✓' if result['success'] else 'FAILED ✗'}")
    print(f"Steps: {result['steps']}")
    print(f"Final Position: {result['final_position']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
