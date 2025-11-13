import numpy as np

# Grid world parameters
GRID_SIZE = 40           # Zero-indexed, non-inclusive
MAXIMUM_TIME_STEPS = 100  # Zero-indexed, non-inclusive, no more than 100 (otherwise edit wind array)

# Velocity bounds (units/s)
VELOCITY_MIN = -3        # Inclusive
VELOCITY_MAX = 3         # Inclusive

# Battery parameters
BATTERY_MIN = 0          # Inclusive
BATTERY_MAX = 95         # Inclusive, no more than MAXIMUM_TIME_STEPS
BATTERY_DRAIN_RATE = 1   # Per time step

# Wind parameters (units/s)
WIND_MIN = -1            # Inclusive, See Below
WIND_MAX = 1             # Inclusive, See Below
# WIND = np.array([[0, 0] for _ in range(100)])
WIND = np.array([
    [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, -1], [1, -1], [1, -1], [0, -1], [0, -1], [0, -1], [-1, -1], [-1, -1], [-1, 0], [-1, 0], [-1, 0], [-1, 1], [-1, 1], [-1, 1], [0, 1],
    [0, 1], [0, 0], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, -1], [1, -1], [0, -1], [0, -1], [-1, -1], [-1, -1], [-1, 0], [-1, 0], [-1, 1], [-1, 1], [0, 1], [0, 0], [0, 1], [0, 1], [1, 1], [1, 0], [1, 0],
    [1, -1], [1, -1], [0, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 0], [1, -1], [1, -1], [0, -1], [0, -1], [-1, -1], [-1, -1], [-1, 0], [-1, 0], [-1, 1],
    [-1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [1, 0], [1, 0], [1, -1], [1, -1], [0, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, -1], [1, -1], [0, -1], [-1, -1], [-1, 0]
])

# Action bounds (acceleration, units/s^2)
ACCEL_MIN = -1           # Inclusive
ACCEL_MAX = 1            # Inclusive

# Obstacle positions (x, y)
# Random Obstacles (Always Random)
OBSTACLES = np.random.randint(0, GRID_SIZE, size=(40, 2))
MAX_OBSTACLES = OBSTACLES.shape[0] # Zero-Indexed, Non-inclusive, See Above

# Drone starting position
STARTING_POSITION = np.array([
    [5, 5]
])

# Goal position
GOAL_POSITION = np.array([35, 35])

# Collision parameters
GOAL_REWARD = 10000
BATTERY_EMPTY_PENALTY = -100
COLLISION_PENALTY = -1000
OUT_OF_BOUNDS_PENALTY = -1000
PROGRESS_REWARD_MULTIPLIER = 100
PROGRESS_REWARD_EPSILON = 0.001
STEP_PENALTY = 0

# Threshold parameters
GOAL_THRESHOLD = 0.25      # Distance threshold to consider goal reached, should not be exactly 0
OBSTACLE_THRESHOLD = 0.50  # Radius around obstacle for collision, should not be exactly 0

# MDP parameters  
DISCOUNT_FACTOR = 0.8
ROLLOUT_NUM_ROLLOUTS = 15
ROLLOUT_MAX_DEPTH = 10

# Plotting parameters
LIVE_UPDATE = True
PLOT_FIGURE_SIZE = 1500