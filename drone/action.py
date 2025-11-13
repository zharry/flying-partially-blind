import sys
sys.path.append("..")

import numpy as np
import config

class DroneAction:
    acceleration: np.ndarray # [ax, ay]

    def __init__(self, acceleration: np.ndarray):
        self.acceleration = acceleration
    
    def __repr__(self):
        return f"DroneAction(acceleration=[{self.acceleration[0]:.2f}, {self.acceleration[1]:.2f}])"
    
    def copy(self):
        return DroneAction(
            acceleration=self.acceleration.copy(),
        )
    
    def set_acceleration(self, a: np.ndarray):
        if a.shape != (2,):
            raise ValueError(f"Invalid acceleration shape: {a.shape}, expected size 2")
        if a[0] < config.ACCEL_MIN or a[0] > config.ACCEL_MAX:
            raise ValueError(f"Invalid acceleration (ax) value: {a[0]}, expected value between {config.ACCEL_MIN} and {config.ACCEL_MAX}")
        if a[1] < config.ACCEL_MIN or a[1] > config.ACCEL_MAX:
            raise ValueError(f"Invalid acceleration (ay) value: {a[1]}, expected value between {config.ACCEL_MIN} and {config.ACCEL_MAX}")
        self.acceleration = a

    def to_array(self) -> np.ndarray:
        return np.array([
            self.acceleration[0], self.acceleration[1]
        ])
    
    @classmethod
    def from_array(cls, a: np.ndarray):
        if a.shape != (2,):
            raise ValueError(f"Invalid acceleration shape: {a.shape}, expected size 2")
        return cls(a)
    
    def to_int(self) -> int:
        # Normalize values to zero-indexed
        ax = int(self.acceleration[0]) - config.ACCEL_MIN
        ay = int(self.acceleration[1]) - config.ACCEL_MIN

        # Compute maximum size for each action
        n_ax = config.ACCEL_MAX - config.ACCEL_MIN + 1
        n_ay = config.ACCEL_MAX - config.ACCEL_MIN + 1

        # Linear indexing
        index = (ax + 
                 ay * n_ax)
        return index
    
    @classmethod
    def from_int(cls, i: int):
        # Compute maximum size for each action
        n_ax = config.ACCEL_MAX - config.ACCEL_MIN + 1
        n_ay = config.ACCEL_MAX - config.ACCEL_MIN + 1

        # Reverse the linear indexing
        ay = i // n_ax
        ax = i % n_ax

        # Denormalize values back to original values
        acceleration = np.array([ax + config.ACCEL_MIN, ay + config.ACCEL_MIN])
        return cls(acceleration=acceleration)

    @staticmethod
    def get_action_space_size() -> int: # Total number of actions, one-indexed, non-inclusive
        return ((config.ACCEL_MAX - config.ACCEL_MIN + 1) *
                (config.ACCEL_MAX - config.ACCEL_MIN + 1))

    @staticmethod
    def get_action_space() -> np.ndarray:
        actions = []
        for ax in range(config.ACCEL_MIN, config.ACCEL_MAX + 1):
            for ay in range(config.ACCEL_MIN, config.ACCEL_MAX + 1):
                actions.append(DroneAction(acceleration=np.array([ax, ay])))
        return np.array(actions)