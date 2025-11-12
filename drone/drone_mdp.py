import sys
sys.path.append("..")

import numpy as np
import config
from drone.state import DroneState
from drone.action import DroneAction

class DroneMDP:
    def transition(self, next_tick: int, state: DroneState, action: DroneAction) -> DroneState:
        if next_tick >= config.MAXIMUM_TIME_STEPS:
            raise ValueError(f"Tick {next_tick} exceeds maximum time steps {config.MAXIMUM_TIME_STEPS}")

        # Create a copy of the state to modify
        next_state = state.copy()
        
        # Update velocity
        new_velocity = state.velocity + action.acceleration + state.wind
        next_state.set_velocity(
            self.clip_vector(new_velocity, config.VELOCITY_MIN, config.VELOCITY_MAX)
        )
        
        # Update position
        previous_position = state.position
        new_position = self.clip_vector(
            state.position + next_state.velocity, 
            0, 
            config.GRID_SIZE - 1
        )
        
        # Check for path intersection with obstacles (on the actual clipped path)
        collision_position = self.check_path_collision(previous_position, new_position)
        if collision_position is not None:
            next_state.set_position(collision_position)
        else:
            next_state.set_position(new_position)
        
        # Update battery
        new_battery = state.battery - config.BATTERY_DRAIN_RATE
        next_state.set_battery(
            self.clip_int(new_battery, config.BATTERY_MIN, config.BATTERY_MAX)
        )
        
        # Update wind
        next_state.set_wind(
            self.clip_vector(
            config.WIND[next_tick],
            config.WIND_MIN,
            config.WIND_MAX)
        )
        
        return next_state
    
    def check_path_collision(self, start_pos: np.ndarray, end_pos: np.ndarray) -> np.ndarray:
        # Check each obstacle
        for obstacle in config.OBSTACLES:
            # Calculate the closest point on the line segment to the obstacle
            closest_point = self.closest_point_on_segment(start_pos, end_pos, obstacle)
            distance = np.linalg.norm(closest_point - obstacle)
            
            # If the path passes through the obstacle threshold
            if distance <= config.OBSTACLE_THRESHOLD:
                # Return the obstacle position (drone stops at obstacle)
                return np.array([obstacle[0], obstacle[1]])
        return None
    
    def closest_point_on_segment(self, start: np.ndarray, end: np.ndarray, point: np.ndarray) -> np.ndarray:
        # Vector from start to end
        segment = end - start
        segment_length_sq = np.dot(segment, segment)
        if segment_length_sq == 0:
            return start
        
        # Project point onto the line, clamped to the segment
        t = np.clip(np.dot(point - start, segment) / segment_length_sq, 0, 1)
        
        # Calculate the closest point on the segment
        closest = start + t * segment
        return closest
    
    def reward(self, state: DroneState, action: DroneAction, next_state: DroneState) -> float:
        # Check if next state is terminal and why
        if next_state.is_terminal():
            if next_state.is_goal_reached():
                return config.GOAL_REWARD
            
            # Collision with obstacle
            if next_state.is_collision():
                return config.COLLISION_PENALTY
            
            # Out of battery
            if next_state.is_out_of_battery():
                return config.BATTERY_EMPTY_PENALTY
        
        # Step penalty
        step_penalty = config.STEP_PENALTY
        
        # Progress reward
        progress_reward = self.calculate_progress_reward(state, next_state)
        
        return step_penalty + progress_reward
    
    def calculate_progress_reward(self, state: DroneState, next_state: DroneState) -> float:
        old_distance = np.linalg.norm(state.position - config.GOAL_POSITION)
        new_distance = np.linalg.norm(next_state.position - config.GOAL_POSITION)
        
        progress = old_distance - new_distance
        return config.PROGRESS_REWARD_MULTIPLIER * progress
    
    @staticmethod
    def clip_int(value: int, min_value: int, max_value: int) -> int:
        return max(min(value, max_value), min_value)

    @staticmethod
    def clip_vector(vector: np.ndarray, min_value: int, max_value: int) -> np.ndarray:
        return np.clip(vector, min_value, max_value)

