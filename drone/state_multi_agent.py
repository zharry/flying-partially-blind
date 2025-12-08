import sys
sys.path.append("..")

import numpy as np
import config
from drone.state import DroneState

class MultiAgentDroneState:
    num_agents: int
    agent_states: list[DroneState]

    def __init__(self, agent_states: list[DroneState]):
        self.num_agents = len(agent_states)
        self.agent_states = agent_states
        
        if self.num_agents != config.NUM_AGENTS:
            raise ValueError(f"Number of agents must be {config.NUM_AGENTS}, got {self.num_agents}")
    
    def __repr__(self):
        states_str = "\n    ".join([f"Agent {i}: {state}" for i, state in enumerate(self.agent_states)])
        return f"MultiAgentState(num_agents={self.num_agents})\n    {states_str}"
    
    def copy(self):
        return MultiAgentDroneState(
            agent_states=[state.copy() for state in self.agent_states]
        )
    
    def get_agent_state(self, agent_id: int) -> DroneState:
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Invalid agent_id: {agent_id}, must be in [0, {self.num_agents})")
        return self.agent_states[agent_id]
    
    def set_agent_state(self, agent_id: int, state: DroneState):
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Invalid agent_id: {agent_id}, must be in [0, {self.num_agents})")
        self.agent_states[agent_id] = state
    
    def is_terminal(self) -> bool:
        # Terminal if any agent has collision or is out of battery
        # Note: We check goal reaching separately below for multi-agent coordination
        for agent_id, state in enumerate(self.agent_states):
            # Check collision with obstacles
            if state.is_collision():
                return True

            # Terminal if any agent collides with any other agents
            for other_id in range(agent_id + 1, len(self.agent_states)):
                if np.linalg.norm(
                    self.agent_states[agent_id].position - self.agent_states[other_id].position) <= config.AGENT_COLLISION_THRESHOLD:
                    return True
            
            # Check battery
            if state.is_out_of_battery():
                return True

        # Check if ALL agents have reached their respective goals
        for agent_id in range(len(self.agent_states)):  
            if config.AGENT_REWARDS[agent_id] != 0:
                return False
        # All agents have reached their goals, so terminal
        return True
