import sys
sys.path.append("..")

import numpy as np

import config
from drone.state import DroneState
from drone.action import DroneAction
from drone.state_multi_agent import MultiAgentDroneState
from drone.action_multi_agent import MultiAgentAction
from mdp.drone import DroneMDP

class MultiAgentDroneMDP:
    agent_mdps: list[DroneMDP]
    num_agents: int

    def __init__(self):
        self.num_agents = config.NUM_AGENTS
        self.agent_mdps = []
        
        # Create agent-specific MDPs with their own A* paths
        # Each agent needs its own MDP because A* path depends on start and goal positions
        original_goal = config.GOAL_POSITION
        original_start = config.STARTING_POSITION
        for agent_id in range(self.num_agents):
            config.GOAL_POSITION = config.MULTI_AGENT_GOAL_POSITIONS[agent_id]
            config.STARTING_POSITION = np.array([config.MULTI_AGENT_STARTING_POSITIONS[agent_id]])
            agent_mdp = DroneMDP()
            self.agent_mdps.append(agent_mdp)
        config.GOAL_POSITION = original_goal
        config.STARTING_POSITION = original_start

    def transition(self, next_tick: int, 
            joint_states: MultiAgentDroneState, 
            joint_actions: MultiAgentAction) -> MultiAgentDroneState:
        if next_tick >= config.MAXIMUM_TIME_STEPS:
            raise ValueError(f"Tick {next_tick} exceeds maximum time steps {config.MAXIMUM_TIME_STEPS}")
        if joint_states.num_agents != self.num_agents:
            raise ValueError(f"Number of agents states must be {self.num_agents}, got {joint_states.num_agents}")
        if joint_actions.num_agents != self.num_agents:
            raise ValueError(f"Number of agents actions must be {self.num_agents}, got {joint_actions.num_agents}")

        # Step 1: ALL agents transition simultaneously (Markov Game property)
        original_positions = []
        tentative_next_states = []
        
        for agent_id in range(self.num_agents):
            agent_state = joint_states.get_agent_state(agent_id)
            agent_action = joint_actions.get_agent_action(agent_id)
            
            # Store original position
            original_positions.append(agent_state.position.copy())
            
            # Use single-agent MDP transition
            tentative_next_state = self.agent_mdps[agent_id].transition(
                next_tick, agent_state, agent_action
            )
            tentative_next_states.append(tentative_next_state)
        
        # Step 2: Check for agent-agent collisions in all pairwise combinations
        # Create a set to track which agents have collisions
        collided_agents = set()
        
        for agent_id in range(self.num_agents):
            for other_id in range(agent_id + 1, self.num_agents):
                # Check if paths cross during this transition
                if self.check_agents_path_collision(
                    original_positions[agent_id], 
                    tentative_next_states[agent_id].position,
                    original_positions[other_id], 
                    tentative_next_states[other_id].position
                ):
                    # Mark both agents as collided
                    collided_agents.add(agent_id)
                    collided_agents.add(other_id)
                
                # Also check if final positions are too close
                elif np.linalg.norm(
                    tentative_next_states[agent_id].position - 
                    tentative_next_states[other_id].position
                ) <= config.AGENT_COLLISION_THRESHOLD:
                    # Mark both agents as collided
                    collided_agents.add(agent_id)
                    collided_agents.add(other_id)
        
        # Step 3: Apply collision resolution
        # Agents that collided stay at their original positions
        final_next_states = []
        for agent_id in range(self.num_agents):
            if agent_id in collided_agents:
                # Collision: revert to original position
                reverted_state = tentative_next_states[agent_id].copy()
                reverted_state.set_position(original_positions[agent_id])
                final_next_states.append(reverted_state)
            else:
                # No collision: use tentative state
                final_next_states.append(tentative_next_states[agent_id])
        
        # Create next joint state
        return MultiAgentDroneState(final_next_states)
    
    def check_agents_path_collision(self, 
            agent_start_pos: np.ndarray, agent_end_pos: np.ndarray,
            other_start_pos: np.ndarray, other_end_pos: np.ndarray) -> bool:
        # Find the minimum distance between the two line segments
        min_distance = self.segment_to_segment_distance(
            agent_start_pos, agent_end_pos,
            other_start_pos, other_end_pos
        )
        
        # Collision if minimum distance is within threshold
        return min_distance <= config.AGENT_COLLISION_THRESHOLD
    
    def segment_to_segment_distance(self,
            p1: np.ndarray, p2: np.ndarray,
            q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compute the minimum distance between two line segments.
        This properly handles all cases including crossing segments.
        
        Args:
            p1, p2: Endpoints of first segment
            q1, q2: Endpoints of second segment
            
        Returns:
            Minimum distance between the two segments
        """
        # Check if segments intersect using cross product method
        def ccw(A, B, C):
            """Check if three points are in counter-clockwise order"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def segments_intersect(A, B, C, D):
            """Check if segment AB and segment CD intersect"""
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        # If segments intersect, they cross each other (distance = 0)
        if segments_intersect(p1, p2, q1, q2):
            return 0.0
        
        # If they don't intersect, minimum distance is one of the endpoint-to-segment distances
        distances = [
            np.linalg.norm(self.agent_mdps[0].closest_point_on_segment(p1, p2, q1) - q1),
            np.linalg.norm(self.agent_mdps[0].closest_point_on_segment(p1, p2, q2) - q2),
            np.linalg.norm(self.agent_mdps[0].closest_point_on_segment(q1, q2, p1) - p1),
            np.linalg.norm(self.agent_mdps[0].closest_point_on_segment(q1, q2, p2) - p2),
        ]
        
        return min(distances)
    
    def reward(self, agent_id: int,
            joint_states: MultiAgentDroneState,
            joint_actions: MultiAgentAction,
            next_joint_state: MultiAgentDroneState) -> float:
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Agent ID must be in [0, {self.num_agents}), got {agent_id}")
        if len(joint_states.agent_states) != self.num_agents:
            raise ValueError(f"Number of agents states must be {self.num_agents}, got {len(joint_states.agent_states)}")
        if len(joint_actions.agent_actions) != self.num_agents:
            raise ValueError(f"Number of agents actions must be {self.num_agents}, got {len(joint_actions.agent_actions)}")
        if next_joint_state.num_agents != self.num_agents:
            raise ValueError(f"Number of next agents states must be {self.num_agents}, got {next_joint_state.num_agents}")

        agent_state = joint_states.get_agent_state(agent_id)
        agent_action = joint_actions.get_agent_action(agent_id)
        next_agent_state = next_joint_state.get_agent_state(agent_id)
        
        # Check for collision with other agents first (highest priority penalty)
        for other_id in range(self.num_agents):
            if other_id == agent_id:
                continue
            # Check both current and next positions for collision
            if (np.linalg.norm(next_agent_state.position - next_joint_state.get_agent_state(other_id).position) 
                <= config.AGENT_COLLISION_THRESHOLD):
                return config.AGENT_COLLISION_PENALTY
        
        # Use single-agent MDP reward for this agent with its specific goal
        # Temporarily override goal for this agent
        original_goal = config.GOAL_POSITION
        config.GOAL_POSITION = config.MULTI_AGENT_GOAL_POSITIONS[agent_id]
        original_reward = config.GOAL_REWARD
        config.GOAL_REWARD = config.AGENT_REWARDS[agent_id]
        reward = self.agent_mdps[agent_id].reward(
            agent_state, agent_action, next_agent_state
        )
        config.GOAL_POSITION = original_goal
        config.GOAL_REWARD = original_reward
        return reward
    
    def get_all_rewards(self, 
            joint_states: MultiAgentDroneState,
            joint_actions: MultiAgentAction,
            next_joint_states: MultiAgentDroneState) -> np.ndarray:
        if joint_states.num_agents != self.num_agents:
            raise ValueError(f"Number of agents states must be {self.num_agents}, got {joint_states.num_agents}")
        if len(joint_actions.agent_actions) != self.num_agents:
            raise ValueError(f"Number of agents actions must be {self.num_agents}, got {len(joint_actions.agent_actions)}")
        if next_joint_states.num_agents != self.num_agents:
            raise ValueError(f"Number of next agents states must be {self.num_agents}, got {next_joint_states.num_agents}")

        rewards = np.zeros(self.num_agents)
        for agent_id in range(self.num_agents):
            rewards[agent_id] = self.reward(
                agent_id, joint_states, joint_actions, next_joint_states
            )
        return rewards

