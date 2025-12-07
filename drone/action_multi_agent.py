import sys
sys.path.append("..")

import numpy as np
import config
from drone.action import DroneAction

class MultiAgentAction:
    num_agents: int
    agent_actions: list[DroneAction]

    def __init__(self, agent_actions: list[DroneAction]):
        self.num_agents = len(agent_actions)
        self.agent_actions = agent_actions

        if self.num_agents != config.NUM_AGENTS:
            raise ValueError(f"Number of agents must be {config.NUM_AGENTS}, got {self.num_agents}")

    def __repr__(self):
        actions_str = "\n    ".join([f"Agent {i}: {action}" for i, action in enumerate(self.agent_actions)])
        return f"MultiAgentAction(num_agents={self.num_agents})\n    {actions_str}"

    def copy(self):
        return MultiAgentAction(
            agent_actions=[action.copy() for action in self.agent_actions]
        )
    
    def get_agent_action(self, agent_id: int) -> DroneAction:
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Invalid agent_id: {agent_id}, must be in [0, {self.num_agents})")
        return self.agent_actions[agent_id]
    
    def set_agent_action(self, agent_id: int, action: DroneAction):
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Invalid agent_id: {agent_id}, must be in [0, {self.num_agents})")
        self.agent_actions[agent_id] = action

