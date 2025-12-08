import sys
sys.path.append("..")

import numpy as np
import config

class DroneState:
    position: np.ndarray  # [x, y]                       current position of the drone
    velocity: np.ndarray  # [vx, vy]                     current velocity of the drone
    battery: int          #                              current battery level of the drone
    wind: np.ndarray      # [wx, wy]                     current wind velocity

    def __init__(self, position: np.ndarray, velocity: np.ndarray, battery: int, wind: np.ndarray):
        self.position = np.asarray(position, dtype=int)
        self.velocity = np.asarray(velocity, dtype=int)
        self.battery = int(battery)
        self.wind = np.asarray(wind, dtype=int)
    
    def __repr__(self):
        return (f"DroneState(position=[{self.position[0]:.2f}, {self.position[1]:.2f}], "
                f"velocity=[{self.velocity[0]:.2f}, {self.velocity[1]:.2f}], "
                f"battery={self.battery:.2f}, "
                f"wind=[{self.wind[0]:.2f}, {self.wind[1]:.2f}])")
    
    def copy(self):
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            battery=self.battery,
            wind=self.wind.copy(),
        )

    def set_position(self, position: np.ndarray):
        if position.shape != (2,):
            raise ValueError(f"Invalid position shape: {position.shape}, expected size 2")
        if position[0] < 0 or position[0] >= config.GRID_SIZE:
            raise ValueError(f"Invalid position (x) value: {position[0]}, expected value between 0 and {config.GRID_SIZE - 1}")
        if position[1] < 0 or position[1] >= config.GRID_SIZE:
            raise ValueError(f"Invalid position (y) value: {position[1]}, expected value between 0 and {config.GRID_SIZE - 1}")
        self.position = np.asarray(position, dtype=int)
    
    def set_velocity(self, velocity: np.ndarray):
        if velocity.shape != (2,):
            raise ValueError(f"Invalid velocity shape: {velocity.shape}, expected size 2")
        if velocity[0] < config.VELOCITY_MIN or velocity[0] > config.VELOCITY_MAX:
            raise ValueError(f"Invalid velocity (vx) value: {velocity[0]}, expected value between {config.VELOCITY_MIN} and {config.VELOCITY_MAX}")
        if velocity[1] < config.VELOCITY_MIN or velocity[1] > config.VELOCITY_MAX:
            raise ValueError(f"Invalid velocity (vy) value: {velocity[1]}, expected value between {config.VELOCITY_MIN} and {config.VELOCITY_MAX}")
        self.velocity = np.asarray(velocity, dtype=int)
    
    def set_battery(self, battery: int):
        if battery < config.BATTERY_MIN or battery > config.BATTERY_MAX:
            raise ValueError(f"Invalid battery value: {battery}, expected value between {config.BATTERY_MIN} and {config.BATTERY_MAX}")
        self.battery = int(battery)
    
    def set_wind(self, wind: np.ndarray):
        if wind.shape != (2,):
            raise ValueError(f"Invalid wind shape: {wind.shape}, expected size 2")
        if wind[0] < config.WIND_MIN or wind[0] > config.WIND_MAX:
            raise ValueError(f"Invalid wind (wx) value: {wind[0]}, expected value between {config.WIND_MIN} and {config.WIND_MAX}")
        if wind[1] < config.WIND_MIN or wind[1] > config.WIND_MAX:
            raise ValueError(f"Invalid wind (wy) value: {wind[1]}, expected value between {config.WIND_MIN} and {config.WIND_MAX}")
        self.wind = np.asarray(wind, dtype=int)
    
    def to_array(self) -> np.ndarray:
        res = np.array([
            self.position[0], self.position[1],
            self.velocity[0], self.velocity[1],
            self.battery,
            self.wind[0], self.wind[1],
        ])
        return res
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        if arr.shape != (7,):
            raise ValueError(f"Invalid array shape: {arr.shape}, expected size 7")
        return cls(
            position=arr[0:2],
            velocity=arr[2:4],
            battery=arr[4],
            wind=arr[5:7],
        )
    
    def to_int(self) -> int:
        # Normalize values to zero-indexed
        x = int(self.position[0])
        y = int(self.position[1])
        vx = int(self.velocity[0]) - config.VELOCITY_MIN
        vy = int(self.velocity[1]) - config.VELOCITY_MIN
        battery = int(self.battery) - config.BATTERY_MIN
        wx = int(self.wind[0]) - config.WIND_MIN
        wy = int(self.wind[1]) - config.WIND_MIN
        
        # Compute maximum size for each state
        n_x = config.GRID_SIZE
        n_y = config.GRID_SIZE
        n_vx = config.VELOCITY_MAX - config.VELOCITY_MIN + 1
        n_vy = config.VELOCITY_MAX - config.VELOCITY_MIN + 1
        n_battery = config.BATTERY_MAX - config.BATTERY_MIN + 1
        n_wx = config.WIND_MAX - config.WIND_MIN + 1
        n_wy = config.WIND_MAX - config.WIND_MIN + 1
        
        # Linear indexing
        index = (x + 
                 y * n_x +
                 vx * (n_x * n_y) +
                 vy * (n_x * n_y * n_vx) +
                 battery * (n_x * n_y * n_vx * n_vy) +
                 wx * (n_x * n_y * n_vx * n_vy * n_battery) +
                 wy * (n_x * n_y * n_vx * n_vy * n_battery * n_wx))
        return index
    
    @classmethod
    def from_int(cls, i: int):
        # Compute maximum size for each state
        n_x = config.GRID_SIZE
        n_y = config.GRID_SIZE
        n_vx = config.VELOCITY_MAX - config.VELOCITY_MIN + 1
        n_vy = config.VELOCITY_MAX - config.VELOCITY_MIN + 1
        n_battery = config.BATTERY_MAX - config.BATTERY_MIN + 1
        n_wx = config.WIND_MAX - config.WIND_MIN + 1
        n_wy = config.WIND_MAX - config.WIND_MIN + 1
        
        # Reverse the linear indexing
        stride_wy = n_x * n_y * n_vx * n_vy * n_battery * n_wx
        wy = i // stride_wy
        i = i % stride_wy
        
        stride_wx = n_x * n_y * n_vx * n_vy * n_battery
        wx = i // stride_wx
        i = i % stride_wx
        
        stride_battery = n_x * n_y * n_vx * n_vy
        battery = i // stride_battery
        i = i % stride_battery
        
        stride_vy = n_x * n_y * n_vx
        vy = i // stride_vy
        i = i % stride_vy
        
        stride_vx = n_x * n_y
        vx = i // stride_vx
        i = i % stride_vx
        
        y = i // n_x
        x = i % n_x
        
        # Denormalize values back to original values
        position = np.array([x, y])
        velocity = np.array([vx + config.VELOCITY_MIN, vy + config.VELOCITY_MIN])
        battery_level = battery + config.BATTERY_MIN
        wind = np.array([wx + config.WIND_MIN, wy + config.WIND_MIN])
        return cls(position=position, velocity=velocity, battery=battery_level, wind=wind)
    
    @staticmethod
    def get_state_space_size() -> int: # Total number of states, one-indexed, non-inclusive
        return (config.GRID_SIZE * config.GRID_SIZE * 
                (config.VELOCITY_MAX - config.VELOCITY_MIN + 1) * 
                (config.VELOCITY_MAX - config.VELOCITY_MIN + 1) *
                (config.BATTERY_MAX - config.BATTERY_MIN + 1) * 
                (config.WIND_MAX - config.WIND_MIN + 1) * 
                (config.WIND_MAX - config.WIND_MIN + 1))

    @staticmethod
    def get_state_space() -> np.ndarray:
        states = []
        for x in range(config.GRID_SIZE):
            for y in range(config.GRID_SIZE):
                for vx in range(config.VELOCITY_MIN, config.VELOCITY_MAX + 1):
                    for vy in range(config.VELOCITY_MIN, config.VELOCITY_MAX + 1):
                        for battery in range(config.BATTERY_MIN, config.BATTERY_MAX + 1):
                            for wx in range(config.WIND_MIN, config.WIND_MAX + 1):
                                for wy in range(config.WIND_MIN, config.WIND_MAX + 1):
                                    states.append(DroneState(position=np.array([x, y]), velocity=np.array([vx, vy]), battery=battery, wind=np.array([wx, wy])))
        return np.array(states)

    def is_goal_reached(self, goal_position: np.ndarray = None) -> bool:
        if goal_position is None:
            goal_position = config.GOAL_POSITION
        if np.linalg.norm(self.position - goal_position) <= config.GOAL_THRESHOLD:
            return True
        return False
    
    def is_collision(self) -> bool:
        for obstacle in config.OBSTACLES:
            if np.linalg.norm(self.position - obstacle) <= config.OBSTACLE_THRESHOLD:
                return True
        return False
        
    def is_out_of_battery(self) -> bool:
        if self.battery <= config.BATTERY_MIN:
            return True
        return False

    def is_terminal(self) -> bool:
        # Goal reached
        if self.is_goal_reached():
            return True
        
        # Collision with any obstacle
        if self.is_collision():
            return True

        # Out of battery
        if self.is_out_of_battery():
            return True
        
        # Otherwise, not terminal
        return False
    
