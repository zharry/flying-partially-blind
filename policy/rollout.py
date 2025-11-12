import random
import sys
sys.path.append("..")

import numpy as np

import config
from mdp.drone import DroneMDP
from drone.state import DroneState
from drone.action import DroneAction

class RolloutPlanner:
    discount: float

    # base_policy is a function that takes state and returns an action, used by the rollout planner
    # num_rollouts is the number of samples to perform for each action to find T(s'|s,a)
    # max_depth is the maximum depth to rollout for each next state
    def __init__(self, mdp: DroneMDP, base_policy, num_rollouts: int, max_depth: int, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.discount = config.DISCOUNT_FACTOR
        self.mdp = mdp
        self.base_policy = base_policy
        self.num_rollouts = num_rollouts
        self.max_depth = max_depth

    # Essentially RolloutLookahead
    def select_action(self, state: DroneState, tick: int) -> tuple[DroneAction, float]:
        # Get all possible actions
        all_actions = DroneAction.get_action_space()
        
        best_action = None
        best_value = -np.inf
        
        # Evaluate each action
        for action in all_actions:
            # Average return over multiple rollouts
            total_value = 0.0
            for _ in range(self.num_rollouts):
                value = self._rollout(state, action, tick)
                total_value += value
            avg_value = total_value / self.num_rollouts
            
            # Track best action
            if avg_value > best_value:
                best_value = avg_value
                best_action = action
        
        # Return best action greedily
        return best_action, best_value
    
    # Essentially the estimated reward plus utility
    def _rollout(self, state: DroneState, first_action: DroneAction, tick: int) -> float:
        current_state = state.copy()
        current_tick = tick
        total_return = 0.0
        discount = 1.0
        
        # Take the first action
        next_state = self.mdp.transition(current_tick, current_state, first_action)
        reward = self.mdp.reward(current_state, first_action, next_state)
        total_return += discount * reward
        current_state = next_state
        current_tick += 1
        discount = self.discount
        depth = 1

        # Continue with base policy
        while (not current_state.is_terminal() and
               depth < self.max_depth):
            
            # Get action from base policy
            action = self.base_policy.select_action(current_state, current_tick, self.rng)
            
            # Take the next action
            next_state = self.mdp.transition(current_tick, current_state, action)
            reward = self.mdp.reward(current_state, action, next_state)
            total_return += discount * reward
            
            # Update for next iteration
            current_state = next_state
            current_tick += 1
            discount *= self.discount
            depth += 1
        
        return total_return


class RandomPolicy:
    # Random Policy uniformly samples from action space
    @staticmethod
    def select_action(state: DroneState, tick: int, rng: np.random.RandomState) -> DroneAction:
        all_actions = DroneAction.get_action_space()
        action_idx = rng.randint(0, len(all_actions))
        return all_actions[action_idx]


class GreedyPolicy:
    # Heuristic Policy that moves towards goal
    @staticmethod
    def select_action(state: DroneState, tick: int, rng: np.random.RandomState) -> DroneAction:
        # Calculate direction to goal
        direction_to_goal = config.GOAL_POSITION - state.position
        desired_acceleration = direction_to_goal - state.velocity - state.wind
        
        # Clip to valid acceleration range
        clipped_acceleration = np.clip(
            desired_acceleration, 
            config.ACCEL_MIN, 
            config.ACCEL_MAX
        ).astype(int)
        
        return DroneAction(clipped_acceleration)

class RandomGreedyPolicy:
    # 50% Chance to select most greedy action
    # 50% Chance to select a random action
    @staticmethod
    def select_action(state: DroneState, tick: int, rng: np.random.RandomState) -> DroneAction:
        choice = random.randint(1, 2)
        if choice == 1:
            return GreedyPolicy.select_action(state, tick, rng)
        else:
            return RandomPolicy.select_action(state, tick, rng)