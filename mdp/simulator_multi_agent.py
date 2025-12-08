import sys
sys.path.append("..")

import numpy as np

import config
from mdp.drone import DroneMDP
from mdp.drone_multi_agent import MultiAgentDroneMDP
from drone.state import DroneState
from drone.action import DroneAction
from drone.state_multi_agent import MultiAgentDroneState
from drone.action_multi_agent import MultiAgentAction

class MultiAgentSimulator:
    def __init__(self, 
                 multi_agent_mdp: MultiAgentDroneMDP,
                 initial_joint_states: MultiAgentDroneState,
                 seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.tick = 0
        self.multi_agent_mdp = multi_agent_mdp
        self.joint_states = initial_joint_states.copy()
        self.num_agents = config.NUM_AGENTS

        if self.joint_states.num_agents != self.num_agents:
            raise ValueError(f"Number of agents states must be {self.num_agents}, got {self.joint_states.num_agents}")
        if self.multi_agent_mdp.num_agents != self.num_agents:
            raise ValueError(f"Number of agents in mdp must be {self.num_agents}, got {self.multi_agent_mdp.num_agents}")
        
        # Track data history
        self.joint_states_history = [self.joint_states.copy()]
        self.joint_actions_history = []
        self.reward_history = []  # [(time_steps, num_agents), ...]
    
    def reset(self, initial_joint_states: MultiAgentDroneState):
        self.tick = 0
        self.joint_states = initial_joint_states.copy()

        if self.joint_states.num_agents != self.num_agents:
            raise ValueError(f"Number of agents states must be {self.num_agents}, got {self.joint_states.num_agents}")
        
        # Clear data history
        self.joint_states_history = [self.joint_states.copy()]
        self.joint_actions_history = []
        self.reward_history = []
    
    def step(self, joint_actions: MultiAgentAction) -> tuple[MultiAgentDroneState, np.ndarray, bool]:
        if self.is_done():
            raise ValueError("Episode already complete. Call reset() to start new episode.")
        
        # Get next state from multi-agent MDP
        next_joint_states = self.multi_agent_mdp.transition(self.tick + 1, self.joint_states, joint_actions)

        # Get rewards for all agents
        rewards = self.multi_agent_mdp.get_all_rewards(self.joint_states, joint_actions, next_joint_states)

        # Check if done
        done = next_joint_states.is_terminal()

        # Stop agents who have reached their goal
        for agent_id in range(self.num_agents):
            if config.AGENT_REWARDS[agent_id] == 0:
                reset_state = next_joint_states.get_agent_state(agent_id).copy()
                reset_state.set_position(config.MULTI_AGENT_GOAL_POSITIONS[agent_id])
                next_joint_states.set_agent_state(agent_id, reset_state)
            if next_joint_states.get_agent_state(agent_id).is_goal_reached(config.MULTI_AGENT_GOAL_POSITIONS[agent_id]):
                config.AGENT_REWARDS[agent_id] = 0
        
        # Update simulator state
        self.joint_states = next_joint_states
        self.tick += 1
        
        # Record history
        self.joint_states_history.append(self.joint_states.copy())
        self.joint_actions_history.append(joint_actions.copy())
        self.reward_history.append(rewards)
        
        return next_joint_states, rewards, done
    
    def is_done(self) -> bool:
        return self.joint_states.is_terminal()
    
    def get_total_reward(self) -> np.ndarray:
        if len(self.reward_history) == 0:
            return np.zeros(self.num_agents)
        return np.sum(self.reward_history, axis=0)

