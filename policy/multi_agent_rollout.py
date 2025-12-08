import sys
sys.path.append("..")

import numpy as np

import config
from mdp.drone_multi_agent import MultiAgentDroneMDP
from mdp.drone import DroneMDP
from drone.state import DroneState
from drone.action import DroneAction
from drone.state_multi_agent import MultiAgentDroneState
from drone.action_multi_agent import MultiAgentAction

class MultiAgentRolloutPlanner:
    discount: float

    # base_policy is a function that takes agent_id, state and returns an action, used by the rollout planner
    # num_rollouts is the number of samples to perform for each joint action (to approximate T[s'|s,a])
    # max_depth is the maximum depth to rollout for each next state
    def __init__(self, mdp: MultiAgentDroneMDP, base_policy, num_rollouts: int, max_depth: int, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.discount = config.DISCOUNT_FACTOR
        self.mdp = mdp
        self.base_policy = base_policy
        self.num_rollouts = num_rollouts
        self.max_depth = max_depth
        self.num_agents = config.NUM_AGENTS

    def select_action(self, state: MultiAgentDroneState, tick: int) -> tuple[MultiAgentAction, np.ndarray]:
        # Get all possible actions for each agent
        single_agent_actions = DroneAction.get_action_space()
        
        # For simplicity, evaluate each agent's action independently
        best_actions = []
        best_values = np.zeros(self.num_agents)
        
        for agent_id in range(self.num_agents):
            agent_state = state.get_agent_state(agent_id)
            best_action = None
            best_value = -np.inf
            
            # Evaluate each possible action for this agent
            for action in single_agent_actions:
                # Create a joint action with this action for current agent
                # and keep other agents' actions from base policy
                test_joint_actions = []
                for aid in range(self.num_agents):
                    if aid == agent_id:
                        test_joint_actions.append(action)
                    else:
                        # Use base policy for other agents
                        other_state = state.get_agent_state(aid)
                        test_joint_actions.append(
                            self.base_policy.select_action(aid, other_state, tick, self.rng)
                        )
                test_joint_action = MultiAgentAction(test_joint_actions)
                
                # Average return over multiple rollouts
                total_value = 0.0
                for _ in range(self.num_rollouts):
                    value = self._rollout(state, test_joint_action, tick, agent_id)
                    total_value += value
                avg_value = total_value / self.num_rollouts
                
                # Track best action for this agent
                if avg_value > best_value:
                    best_value = avg_value
                    best_action = action
            
            best_actions.append(best_action)
            best_values[agent_id] = best_value
        
        # Return best joint action greedily
        return MultiAgentAction(best_actions), best_values
    
    def _rollout(self, state: MultiAgentDroneState, first_joint_action: MultiAgentAction, tick: int, agent_id: int) -> float:
        current_state = state.copy()
        current_tick = tick
        total_return = 0.0
        discount = 1.0
        depth = 0
        
        while (not current_state.is_terminal() and
               depth < self.max_depth):
            if depth == 0:
                joint_action = first_joint_action
            else:
                # Use base policy for all agents
                agent_actions = []
                for aid in range(self.num_agents):
                    agent_state = current_state.get_agent_state(aid)
                    action = self.base_policy.select_action(aid, agent_state, current_tick, self.rng)
                    agent_actions.append(action)
                joint_action = MultiAgentAction(agent_actions)
            
            # Take the joint action
            next_state = self.mdp.transition(current_tick + 1, current_state, joint_action)
            reward = self.mdp.reward(agent_id, current_state, joint_action, next_state)
            total_return += discount * reward
            current_state = next_state
            current_tick += 1
            discount *= self.discount
            depth += 1
        
        return total_return


class AStarPolicy:
    def __init__(self):
        # Store MDPs for each agent with their specific goals
        self.agent_mdps = []
        
        # Store original config values
        original_goal = config.GOAL_POSITION
        original_start = config.STARTING_POSITION
        
        for agent_id in range(config.NUM_AGENTS):
            # Set agent-specific goal and start
            config.GOAL_POSITION = config.MULTI_AGENT_GOAL_POSITIONS[agent_id]
            config.STARTING_POSITION = np.array([config.MULTI_AGENT_STARTING_POSITIONS[agent_id]])
            
            # Create a separate MDP for this agent to compute the A* path for this agent's goal
            print(f"Creating A* path for Agent {agent_id}:")
            print(f"  Start: {config.STARTING_POSITION[0]}, Goal: {config.GOAL_POSITION}")
            print(f"  Obstacles shape: {config.OBSTACLES.shape}, Clearance: {config.ASTAR_OBSTACLE_CLEARANCE}")
            agent_mdp = DroneMDP()
            print(f"  A* path length: {len(agent_mdp.shortest_path)}, Path distance: {agent_mdp.shortest_path_distance:.2f}")
            print()
            self.agent_mdps.append(agent_mdp)
        
        # Restore original config values
        config.GOAL_POSITION = original_goal
        config.STARTING_POSITION = original_start
    
    def select_action(self, agent_id: int, state: DroneState, tick: int, rng: np.random.RandomState) -> DroneAction:
        # Get the A* path for this specific agent
        mdp = self.agent_mdps[agent_id]
        path = mdp.shortest_path
        
        # Find the closest reachable point on the path using BFS
        _, distance_along_path = mdp._astar_path_get_closest_point_on_path_with_bfs(state.position)
        
        # Convert distance_along_path to a path index
        cumulative_distance = 0.0
        closest_idx = 0
        for i in range(len(path) - 1):
            segment_length = np.linalg.norm(path[i + 1] - path[i])
            if cumulative_distance + segment_length >= distance_along_path:
                closest_idx = i
                break
            cumulative_distance += segment_length
        else:
            # If we've gone through all segments, we're at the end
            closest_idx = len(path) - 1
        
        # Look ahead on the path (target a waypoint ahead of our closest point)
        lookahead_distance = config.ASTAR_LOOKAHEAD_DISTANCE
        target_idx = min(closest_idx + lookahead_distance, len(path) - 1)
        target_waypoint = path[target_idx]
        
        # Calculate direction to target waypoint
        direction_to_target = target_waypoint - state.position
        distance = np.linalg.norm(direction_to_target)

        if distance <= 0:
            return DroneAction(np.array([0, 0]))

        # Calculate desired velocity to reach target
        direction_unit = direction_to_target / distance
        desired_velocity = direction_unit * min(config.VELOCITY_MAX, distance)
        
        # Calculate desired acceleration accounting for current velocity and wind
        desired_acceleration = desired_velocity - state.velocity - state.wind
        
        # Clip to valid acceleration range and round
        clipped_acceleration = np.clip(
            desired_acceleration,
            config.ACCEL_MIN,
            config.ACCEL_MAX
        )
        rounded_acceleration = np.round(clipped_acceleration).astype(int)
        
        return DroneAction(rounded_acceleration)

