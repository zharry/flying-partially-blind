from __future__ import annotations
import sys
sys.path.append("..")

import heapq
import math
import pomdp_py
import random
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

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

# Note: Starting position and goal are now defined per test case in config.py

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
    
    Probability model:
    - Unobserved cells: 0% obstacle probability (optimistically assumed free)
    - First observation: 80% confidence in what was observed
      (OCCUPIED reading → 80%, FREE reading → 20%)
    - Subsequent observations: Bayesian update with 80% sensor accuracy
    
    Optimized for incremental updates:
    - Only tracks cells that have been observed
    - Tracks which cells changed in the most recent observation
    """
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # Default log odds is 0 (Probability 0.5)
        # Key is Coordinate, value is log-odds float
        self.grid_log_odds: Dict[Coordinate, float] = defaultdict(float)
        
        # Optimization: Track observed cells for sparse sampling
        self.observed_cells: set = set()  # All cells ever observed
        self.last_updated_cells: set = set()  # Cells updated in current observation
        
    def begin_observation(self) -> None:
        """Call before processing a new observation to reset change tracking."""
        self.last_updated_cells.clear()

    def update(self, coord: Coordinate, is_observed_occupied: bool) -> None:
        """
        Bayesian update of the cell using Log-Odds.
        
        First observation: Sets probability to 80% confidence in observation.
        - Observed as OCCUPIED → 80% obstacle probability
        - Observed as FREE → 20% obstacle probability
        
        Subsequent observations: Bayesian update with 80% sensor accuracy.
        """
        # Track this cell as updated in this observation cycle
        self.last_updated_cells.add(coord)
        
        # Check if this is the first observation for this cell
        is_first_observation = coord not in self.observed_cells
        self.observed_cells.add(coord)
        
        if is_first_observation:
            # First observation: directly SET to sensor confidence level
            if is_observed_occupied:
                self.grid_log_odds[coord] = LO_OCCUPIED  # P(occupied) = 0.8
            else:
                self.grid_log_odds[coord] = LO_FREE      # P(occupied) = 0.2
        else:
            # Subsequent observations: Bayesian update (ADD to existing)
            if is_observed_occupied:
                self.grid_log_odds[coord] += LO_OCCUPIED
            else:
                self.grid_log_odds[coord] += LO_FREE
            
        # Clamp values to prevent infinity
        current = self.grid_log_odds[coord]
        self.grid_log_odds[coord] = max(min(current, MAX_CONFIDENCE), -MAX_CONFIDENCE)
    
    def get_probability(self, coord: Coordinate) -> float:
        """
        Get probability of cell being occupied.
        
        - Unobserved cells: 0% probability (optimistically assumed free)
        - First observation: 80% confidence in what was observed
          (FREE → 20% occupied, OCCUPIED → 80% occupied)
        - Subsequent observations: Bayesian-updated probability
        """
        if coord not in self.observed_cells:
            return 0.0  # Unobserved = optimistically assumed free
        l_val = self.grid_log_odds[coord]
        return 1.0 - (1.0 / (1.0 + math.exp(l_val)))

    def sample_map_hypothesis(self) -> List[Coordinate]:
        """
        Generates a concrete map (list of obstacles) for a particle
        based on current probabilities.
        
        OPTIMIZED: Only samples from observed cells (sparse sampling).
        Unobserved cells have 0% probability — assumed free (no obstacles).
        """
        generated_obstacles: List[Coordinate] = []
        
        # Only iterate over cells we've actually observed
        # Unobserved cells = 0% obstacle probability, so they're never added
        for coord in self.observed_cells:
            prob_occupied = self.get_probability(coord)
            
            # Roll the dice for this particle
            if random.random() < prob_occupied:
                generated_obstacles.append(coord)
                    
        return generated_obstacles
    
    def sample_incremental(self, parent_obstacles: frozenset) -> List[Coordinate]:
        """
        FAST incremental sampling: Only resample cells that changed in this observation.
        
        Args:
            parent_obstacles: Obstacle set from a parent particle to inherit from
            
        Returns:
            New obstacle list with only the changed cells resampled
        """
        # Start with parent's obstacles (as a mutable set)
        new_obstacles = set(parent_obstacles)
        
        # Only resample the cells that were updated in this observation
        for coord in self.last_updated_cells:
            prob_occupied = self.get_probability(coord)
            
            if random.random() < prob_occupied:
                new_obstacles.add(coord)
            else:
                new_obstacles.discard(coord)
        
        return list(new_obstacles)

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

class AStarPathfinder:
    """
    A* pathfinder with caching based on obstacle configuration.
    
    Cache key: (obstacle_frozenset, start_coord, goal_coord)
    Cache value: (steps_to_goal, next_coord, next_action)
    """
    
    def __init__(self, width: int, height: int, actions: List[GridAction]):
        self.width = width
        self.height = height
        # Only use movement actions (exclude 'stay')
        self.movement_deltas = [(a.delta, a) for a in actions if a.delta != (0, 0)]
        # Cache: (obstacle_frozenset, start_coord, goal_coord) -> (steps, next_coord, next_action)
        self._cache: Dict[Tuple[frozenset, Coordinate, Coordinate], Tuple[int, Optional[Coordinate], Optional[GridAction]]] = {}
    
    def _heuristic(self, coord: Coordinate, goal: Coordinate) -> float:
        """Octile distance (diagonal moves cost sqrt(2), cardinal moves cost 1)."""
        dx = abs(coord.x - goal.x)
        dy = abs(coord.y - goal.y)
        # Octile distance: move diagonally as much as possible, then straight
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
    
    def find_path(self, start: Coordinate, goal: Coordinate,
                  obstacle_set: frozenset) -> Tuple[int, Optional[Coordinate], Optional[GridAction]]:
        """
        Find shortest path using A*.
        
        Args:
            start: Starting coordinate
            goal: Goal coordinate
            obstacle_set: Frozenset of obstacle coordinates
            
        Returns:
            (steps_to_goal, next_node, action_to_next_node)
            - If no path exists: (infinity as large int, None, None)
            - If already at goal: (0, start, None)
        """
        # Check cache
        cache_key = (obstacle_set, start, goal)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Already at goal
        if start == goal:
            result = (0, start, None)
            self._cache[cache_key] = result
            return result
        
        # A* search
        # Priority queue: (f_score, counter, coord)
        counter = 0
        open_set = [(self._heuristic(start, goal), counter, start)]
        came_from: Dict[Coordinate, Tuple[Coordinate, GridAction]] = {}  # node -> (parent, action)
        g_score = {start: 0}
        visited = set()
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                # Reconstruct path to find the first step
                path = []
                node = current
                while node in came_from:
                    parent, action = came_from[node]
                    path.append((node, action))
                    node = parent
                
                # path[-1] is (first_node_after_start, action_from_start)
                if path:
                    steps = len(path)
                    next_node, next_action = path[-1]
                    result = (steps, next_node, next_action)
                else:
                    result = (0, start, None)
                
                self._cache[cache_key] = result
                return result
            
            for delta, action in self.movement_deltas:
                nx = current.x + delta[0]
                ny = current.y + delta[1]
                
                # Check bounds
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                
                neighbor = Coordinate(nx, ny)
                
                # Check obstacle
                if neighbor in obstacle_set:
                    continue
                
                if neighbor in visited:
                    continue
                
                # Diagonal moves cost sqrt(2), cardinal moves cost 1
                move_cost = math.sqrt(2) if (delta[0] != 0 and delta[1] != 0) else 1.0
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, action)
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))
        
        # No path found - use large penalty value
        result = (10000, None, None)
        self._cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """Clear the pathfinding cache."""
        self._cache.clear()


class GridRewardModel(pomdp_py.RewardModel):
    """
    A*-based reward model.
    
    Reward structure:
    - Reaching the goal: +100 (terminal reward)
    - Collision (tried to move but stayed): -100
    - Otherwise: negative of A* steps from next_state to goal
    - Stay action (not at goal): additional -1 penalty
    """
    
    def __init__(self, pathfinder: AStarPathfinder, no_path_penalty: float = -100.0,
                 goal_reward: float = 100.0):
        """
        Args:
            pathfinder: AStarPathfinder instance for computing distances
            no_path_penalty: Penalty when no path to goal exists
            goal_reward: Bonus reward for reaching the goal
        """
        self.pathfinder = pathfinder
        self.no_path_penalty = no_path_penalty
        self.goal_reward = goal_reward
    
    def sample(self, state: GridState, action: GridAction, next_state: GridState) -> float:
        # Check if we reached the goal - give large positive reward
        if next_state.drone_pos == next_state.goal:
            return self.goal_reward
        
        # Check if drone tried to move but couldn't (collision with wall/obstacle)
        if state.drone_pos == next_state.drone_pos and action.delta != (0, 0):
            return -100.0
        
        # Run A* from next_state to goal
        steps, _, _ = self.pathfinder.find_path(
            next_state.drone_pos,
            next_state.goal,
            next_state._obstacle_set
        )
        
        # If no path exists (steps is very large), return large penalty
        if steps >= 10000:
            return self.no_path_penalty
        
        # Base reward is negative steps
        reward = -float(steps)
        
        # Stay action gets additional -1 penalty (only when not at goal)
        if action.delta == (0, 0):
            reward -= 1.0
        
        return reward

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
    """
    A*-based rollout policy.
    
    Returns the action that A* recommends as the next step toward the goal.
    This provides an informed rollout that follows the optimal path given
    the current obstacle hypothesis.
    """
    
    def __init__(self, actions, pathfinder: AStarPathfinder):
        """
        Args:
            actions: List of available actions
            pathfinder: AStarPathfinder instance for computing paths
        """
        self.actions = actions
        self.pathfinder = pathfinder
    
    def rollout(self, state, history=None):
        """
        Return the action that A* suggests for reaching the goal.
        
        If no path exists or already at goal, returns 'stay' action.
        """
        steps, next_node, next_action = self.pathfinder.find_path(
            state.drone_pos,
            state.goal,
            state._obstacle_set
        )
        
        # If A* found a valid action, use it
        if next_action is not None:
            return next_action
        
        # If no path or already at goal, return stay action
        return ACTIONS_DICT.get("stay", random.choice(self.actions))