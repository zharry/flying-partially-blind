from __future__ import annotations
import pomdp_py
from typing import Tuple, List, Dict
import random
from collections import defaultdict
import math

from config import (
    GRID_WIDTH, GRID_HEIGHT, SENSOR_ACCURACY, SENSOR_RANGE,
    LO_OCCUPIED, LO_FREE, MAX_CONFIDENCE, DIRECTIONS
)

class Coordinate(): 
    def __init__(self, x:int , y:int) -> None:
        self.x = x
        self.y = y
    def __str__(self) -> str:
        return str(self.x)+"-"+str(self.y)
    def __repr__(self) -> str:
        return f"Coordinate({self.x}, {self.y})"
    def __eq__(self, other: Coordinate) -> bool:
        if not isinstance(other, Coordinate):
            return False
        return self.x == other.x and self.y == other.y
    def __hash__(self) -> int:
        return hash((self.x, self.y))

# Define constants after Coordinate class
ROBOT_STARTING_POSITION = Coordinate(0, 0)
ROBOT_GOAL_POSITION = Coordinate(9, 9)

class GridState(pomdp_py.State):
    
    def __init__(self, drone_pos: Coordinate, obstacle_pos: List[Coordinate], goal: Coordinate) -> None:
        self.drone_pos = drone_pos
        self.obstacle_pos = obstacle_pos
        self._obstacle_set = frozenset(obstacle_pos)  # For O(1) lookup
        self.goal = goal

    def __hash__(self) -> int:
        hash_str = str(self.drone_pos)
        hash_str += str(self.goal)
        # Sort obstacles to ensure consistent hash regardless of list order
        for obstacle in sorted(self.obstacle_pos, key=lambda c: (c.x, c.y)): 
            hash_str += str(obstacle)
        return hash(hash_str)
    
    def __eq__(self, other: GridState) -> bool:
        if not isinstance(other, GridState):
            return False
        if self.drone_pos != other.drone_pos or self.goal != other.goal:
            return False
        if len(self.obstacle_pos) != len(other.obstacle_pos):
            return False
        # Explicitly check if each obstacle position is in the other list
        for obstacle in self.obstacle_pos:
            if obstacle not in other.obstacle_pos:
                return False
        return True

class LidarObservation(pomdp_py.Observation):
    """
    beams: A tuple of 8 results. 
    Each result is a tuple of length up to SENSOR_RANGE.
    0 = Free, 1 = Blocked, None = Out of bounds/Unknown
    Example: ((0, 1), (0, 0), ...) means North is Free then Blocked.
    """
    def __init__(self, beams) -> None:
        self.beams = tuple(beams)
    def __hash__(self): return hash(self.beams)
    def __eq__(self, other): return self.beams == other.beams

class GlobalOccupancyGrid:
    """
    The persistent memory of the robot.
    Stores Log-Odds for every cell.
    """
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # Default log odds is 0 (Probability 0.5)
        # Key is Coordinate, value is log-odds float
        self.grid_log_odds: Dict[Coordinate, float] = defaultdict(float) 

    def update(self, coord: Coordinate, is_observed_occupied: bool) -> None:
        """
        Bayesian update of the cell using Log-Odds.
        """
        # 1. Update value
        if is_observed_occupied:
            self.grid_log_odds[coord] += LO_OCCUPIED
        else:
            self.grid_log_odds[coord] += LO_FREE
            
        # 2. Clamp values to prevent infinity
        current = self.grid_log_odds[coord]
        self.grid_log_odds[coord] = max(min(current, MAX_CONFIDENCE), -MAX_CONFIDENCE)

    def sample_map_hypothesis(self) -> List[Coordinate]:
        """
        Generates a concrete map (list of obstacles) for a particle
        based on current probabilities.
        """
        generated_obstacles: List[Coordinate] = []
        
        # We assume the world is the size of the grid.
        # Ideally, we iterate over all 'known' cells or the whole bounds.
        for x in range(self.width):
            for y in range(self.height):
                coord = Coordinate(x, y)
                l_val = self.grid_log_odds.get(coord, 0.0) # Default 0.0 (50/50)
                
                # Convert Log-Odds back to Probability
                # P = 1 - (1 / (1 + e^L))
                prob_occupied = 1.0 - (1.0 / (1.0 + math.exp(l_val)))
                
                # Roll the dice for this particle
                if random.random() < prob_occupied:
                    generated_obstacles.append(coord)
                    
        return generated_obstacles

class GridAction(pomdp_py.Action):
    def __init__(self, name: str, delta: Tuple[int, int]) -> None:
        self.name = name
        self.delta = delta  # (dx, dy) for movement
    def __hash__(self): return hash(self.name)
    def __eq__(self, other): return self.name == other.name
    def __repr__(self): return f"Action({self.name})"

# Pre-defined actions with their movement deltas
ACTIONS = [
    GridAction("north", (0, 1)),
    GridAction("south", (0, -1)),
    GridAction("east", (1, 0)),
    GridAction("west", (-1, 0)),
    GridAction("northeast", (1, 1)),
    GridAction("northwest", (-1, 1)),
    GridAction("southeast", (1, -1)),
    GridAction("southwest", (-1, -1)),
    GridAction("stay", (0, 0)),
]
ACTIONS_DICT = {a.name: a for a in ACTIONS}

class GridObservationModel(pomdp_py.ObservationModel):
    def __init__(self, directions):
        self.directions = directions

    def probability(self, observation, next_state, action):
        """
        P(o | s', a).
        Used by the planner to weigh particles.
        """
        # We simulate what the observation SHOULD be for this state
        expected_beams = self._raycast(next_state)
        
        prob = 1.0
        
        # Compare Actual vs Expected
        for i, real_beam in enumerate(observation.beams):
            expected_beam = expected_beams[i]
            
            # Simple approach: Check step by step
            for depth in range(len(real_beam)):
                real_val = real_beam[depth]
                exp_val = expected_beam[depth]
                
                if real_val == exp_val:
                    prob *= SENSOR_ACCURACY
                else:
                    prob *= (1.0 - SENSOR_ACCURACY)
                    
        return prob

    def sample(self, next_state: GridState, action: GridAction) -> LidarObservation:
        """
        Used by Planner to simulate observations in the tree.
        """
        true_beams = self._raycast(next_state)
        noisy_beams = []
        
        for beam in true_beams:
            noisy_beam = []
            for val in beam:
                # Apply noise
                if random.random() < SENSOR_ACCURACY:
                    noisy_beam.append(val)
                else:
                    # Flip the bit (0->1, 1->0)
                    noisy_beam.append(1 - val)
            noisy_beams.append(tuple(noisy_beam))
            
        return LidarObservation(tuple(noisy_beams))

    def _raycast(self, state: GridState):
        """
        Helper to calculate perfect vision for a state.
        Returns list of 8 beams. Each beam is tuple e.g. (0, 1)
        """
        beams = []
        rx, ry = state.drone_pos.x, state.drone_pos.y
        
        for dx, dy in self.directions:
            beam_data = []
            curr_x, curr_y = rx, ry
            
            for _ in range(SENSOR_RANGE):
                curr_x += dx
                curr_y += dy
                
                # Check bounds
                if not (0 <= curr_x < GRID_WIDTH and 0 <= curr_y < GRID_HEIGHT):
                    beam_data.append(1) # Treat wall as obstacle
                    continue
                
                # Check if this coordinate is in obstacles (O(1) lookup)
                curr_coord = Coordinate(curr_x, curr_y)
                if curr_coord in state._obstacle_set:
                    beam_data.append(1) # Blocked
                else:
                    beam_data.append(0) # Free
            
            beams.append(tuple(beam_data))
        return beams

class GridTransitionModel(pomdp_py.TransitionModel):
    def sample(self, state: GridState, action: GridAction) -> GridState:
        # Deterministic movement for the robot itself
        # But map stays static
        nx = state.drone_pos.x + action.delta[0]
        ny = state.drone_pos.y + action.delta[1]
        
        new_pos = Coordinate(nx, ny)
        
        # Check collision with THIS particle's map hypothesis
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
            if new_pos not in state._obstacle_set:
                return GridState(new_pos, state.obstacle_pos, state.goal)
        
        # Collision or out of bounds -> Stay still
        return GridState(state.drone_pos, state.obstacle_pos, state.goal)

class GridRewardModel(pomdp_py.RewardModel):
    def __init__(self, progress_weight: float = 2.0):
        """
        Args:
            progress_weight: Multiplier for distance-based progress reward
        """
        self.progress_weight = progress_weight
    
    def _manhattan_distance(self, pos: Coordinate, goal: Coordinate) -> int:
        return abs(pos.x - goal.x) + abs(pos.y - goal.y)
    
    def sample(self, state: GridState, action: GridAction, next_state: GridState) -> float:
        # Check if drone reached the goal
        if next_state.drone_pos == next_state.goal:
            return 100.0
        
        # Check if drone tried to move but couldn't (collision with wall/obstacle)
        if state.drone_pos == next_state.drone_pos and action.delta != (0, 0):
            return -100.0
        
        # Stay action gets flat penalty
        if action.delta == (0, 0):
            return -1.0
        
        # Progress-based reward: reward for getting closer to goal
        dist_before = self._manhattan_distance(state.drone_pos, state.goal)
        dist_after = self._manhattan_distance(next_state.drone_pos, next_state.goal)
        progress = dist_before - dist_after  # Positive if closer, negative if farther
        
        # Base step cost + progress reward
        return -1.0 + self.progress_weight * progress

class GridPolicyModel(pomdp_py.PolicyModel):
    """Policy model that returns all valid actions for POMCP."""
    def __init__(self, actions):
        self.actions = tuple(actions)  # Must be tuple for Cython compatibility
    
    def get_all_actions(self, state=None, history=None):
        """Return all possible actions (required by POMCP)."""
        return self.actions
    
    def sample(self, state, history=None):
        """Sample a random action (used for rollouts)."""
        return random.choice(self.actions)

class GridRolloutPolicy(pomdp_py.RolloutPolicy):
    """Custom rollout policy that returns the action with highest immediate reward."""
    def __init__(self, actions, transition_model: GridTransitionModel, reward_model: GridRewardModel):
        self.actions = actions
        self.transition_model = transition_model
        self.reward_model = reward_model
    
    def rollout(self, state, history):
        """Return the action that maximizes immediate reward (greedy rollout)."""
        best_action = None
        best_reward = float('-inf')
        
        for action in self.actions:
            # Simulate the next state
            next_state = self.transition_model.sample(state, action)
            # Calculate reward for this action
            reward = self.reward_model.sample(state, action, next_state)
            
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        return best_action if best_action is not None else random.choice(self.actions)