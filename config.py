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
# Random Obstacles
# OBSTACLES = np.array([ [ 8,  8], [28, 30], [20, 33], [15, 27], [32, 13], [23,  9], [35, 30], [13,  5], [19, 28], [ 7, 22], [35, 21], [21, 19], [ 4, 10], [33, 27], [ 6, 16], [26, 24], [ 35, 29], [17, 11], [31, 33], [10, 32], [29, 10], [30,  6], [27, 35], [24, 17], [18,  8], [25, 13], [ 5, 31], [22, 14], [33,  7], [ 6, 26], [35,  5], [11, 30], [16, 20], [34, 21], [ 8, 16], [28, 33], [13, 26], [ 7,  8], [24,  9], [19, 17]])
# OBSTACLES = np.array([[14, 22], [7, 36], [33, 8], [25, 19], [19, 11], [37, 2], [31, 29], [22, 5], [29, 14], [9, 27], [11, 35], [34, 21], [16, 6], [36, 17], [27, 24], [5, 33], [12, 37], [24, 9], [30, 30], [15, 15], [32, 28], [8, 12], [26, 26], [17, 23], [13, 18], [23, 7], [38, 32], [28, 10], [21, 13], [35, 20], [20, 16], [6, 34], [18, 37], [10, 31], [38, 25], [3, 22], [4, 14], [1, 38], [2, 21], [36, 11]])

# 2-Line Maze Obstacles
OBSTACLES = np.array([[x, 12] for x in range(0, 25)] + [[x, 25] for x in range(15, 40)])
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
ROLLOUT_NUM_ROLLOUTS = 40
ROLLOUT_MAX_DEPTH = 40

# Plotting parameters
LIVE_UPDATE = True
PLOT_FIGURE_SIZE = 1500