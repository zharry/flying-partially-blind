from state_observation import *
from agent import *
from config import DIRECTIONS, GRID_WIDTH, GRID_HEIGHT
from visualization import GridVisualizer
import pomdp_py

def main():
    # Init Agent (uses DIRECTIONS and ACTIONS from config/state_observation)
    agent = RobotAgent(ROBOT_STARTING_POSITION, ROBOT_GOAL_POSITION, DIRECTIONS)
    
    # Global history of visited positions (shared with rollout policy)
    global_visited = set()
    
    # Init Planner (POMCP)
    # POMCP is technically POUCT applied to POMDPs.
    # Note: planner.update() updates both tree and belief, but we overwrite
    # the belief manually after for more control.
    # Use greedy rollout policy with revisit penalty
    transition_model = GridTransitionModel()
    reward_model = GridRewardModel()
    rollout_policy = GridRolloutPolicy(
        ACTIONS, transition_model, reward_model,
        global_visited=global_visited,
        revisit_penalty=-10.0
    )
    planner = pomdp_py.POMCP(max_depth=10, exploration_const=5.0, num_sims=500, 
                              rollout_policy=rollout_policy)

    print("Starting Simulation...")
    
    # Mock Real World (The Truth)
    true_obstacles = [Coordinate(2, 2), Coordinate(2, 3), Coordinate(2, 4), Coordinate(5, 5)]
    true_robot_pos = ROBOT_STARTING_POSITION
    
    # Add starting position to visited set
    global_visited.add(true_robot_pos)
    
    # Initialize visualizer
    viz = GridVisualizer(
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        obstacles=true_obstacles,
        goal=ROBOT_GOAL_POSITION
    )
    viz.update(true_robot_pos, step=0, action_name="start")
    
    for step in range(200):
        # 1. Plan
        action = planner.plan(agent)
        print(f"Step {step}: Moving {action.name}")
        
        # 2. Execute (Update True World)
        # (Simple physics for simulation)
        nx = true_robot_pos.x + action.delta[0]
        ny = true_robot_pos.y + action.delta[1]
        new_pos = Coordinate(nx, ny)
        
        if new_pos in true_obstacles or not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
            print("  -> Bonk! Hitting wall.")
        else:
            true_robot_pos = new_pos
            # Track visited position
            global_visited.add(true_robot_pos)
            
        print(f"  -> True Pos: {true_robot_pos} (visited: {len(global_visited)} positions)")
        
        # Update visualization
        viz.update(true_robot_pos, step=step+1, action_name=action.name)
        
        # 3. Observe (Generate 80% accurate observation)
        # We cheat and use the agent's observation model to generate a sample
        # based on the TRUE state.
        true_state = GridState(true_robot_pos, true_obstacles, ROBOT_GOAL_POSITION)
        real_obs = agent.observation_model.sample(true_state, action)
        
        # 4. MANUAL BELIEF UPDATE (completely bypasses POMCP's belief filtering)
        # We DON'T call planner.update() because it does particle filtering which
        # causes deprivation. Instead, we just set the belief directly.
        # POMCP will rebuild the tree from scratch using our belief on next plan().
        agent.manual_belief_update(action, real_obs, true_robot_pos)
        
        # 5. Reset tree so POMCP starts fresh with the new belief
        # This prevents issues where the old tree doesn't match the new belief
        agent.tree = None
        
        # Check Goal
        if true_robot_pos == ROBOT_GOAL_POSITION:
            print("Goal Reached!")
            viz.show_goal_reached()
            break
    
    # Keep window open at the end
    viz.show()

if __name__ == "__main__":
    main()