import numpy as np

import config
from drone.drone_mdp import DroneMDP
from drone.state import DroneState
from drone.action import DroneAction

class Simulator:
    def __init__(self, mdp: DroneMDP, initial_state: DroneState, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.tick = 0
        self.mdp = mdp
        self.state = initial_state.copy()
        
        # Track data history
        self.state_history = [self.state.copy()]
        self.action_history = []
        self.reward_history = []
    
    def reset(self, initial_state: DroneState):
        self.tick = 0
        self.state = initial_state.copy()
        
        # Clear data history
        self.state_history = [self.state.copy()]
        self.action_history = []
        self.reward_history = []
    
    def step(self, action: DroneAction) -> tuple[DroneState, float, bool]:
        if self.is_done():
            raise ValueError("Episode already complete. Call reset() to start new episode.")
        
        # Get next state from MDP
        next_state = self.mdp.transition(self.tick + 1, self.state, action)
        
        # Get reward
        reward = self.mdp.reward(self.state, action, next_state)
        
        # Check if done
        done = next_state.is_terminal()
        
        # Update state and tick
        self.state = next_state
        self.tick += 1
        
        # Record history
        self.state_history.append(self.state.copy())
        self.action_history.append(action.copy())
        self.reward_history.append(reward)
        
        return next_state, reward, done
    
    def is_done(self) -> bool:
        return self.state.is_terminal()
    
    def get_total_reward(self) -> float:
        return sum(self.reward_history)
