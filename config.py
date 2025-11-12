import numpy as np

# Grid world parameters
GRID_SIZE = 40           # Zero-indexed, non-inclusive
MAXIMUM_TIME_STEPS = 35  # Zero-indexed, non-inclusive, no more than 100 (otherwise edit wind array)

# Velocity bounds (units/s)
VELOCITY_MIN = -3        # Inclusive
VELOCITY_MAX = 3         # Inclusive

# Battery parameters
BATTERY_MIN = 0          # Inclusive
BATTERY_MAX = 30         # Inclusive, no more than MAXIMUM_TIME_STEPS
BATTERY_DRAIN_RATE = 1   # Per time step

# Wind parameters (units/s)
WIND_MIN = -1            # Inclusive, See Below
WIND_MAX = 1             # Inclusive, See Below
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
MAX_OBSTACLES = 40       # Zero-Indexed, Non-inclusive, See Below
# Random Obstacles
OBSTACLES = np.array([ [ 8,  7], [28, 30], [20, 33], [15, 27], [32, 13], [23,  9], [14, 35], [13,  5], [19, 28], [ 7, 22], [35, 21], [21, 19], [ 5, 12], [12, 29], [ 6, 16], [26, 24], [ 9, 25], [17, 11], [31, 33], [10, 32], [29, 10], [30,  6], [27, 34], [24, 17], [18,  8], [25, 13], [ 5, 31], [22, 14], [33,  7], [ 6, 26], [35,  5], [11, 30], [16, 20], [34, 21], [ 8, 16], [28, 33], [13, 26], [ 7,  8], [24,  9], [19, 17]])

# Drone starting position
STARTING_POSITION = np.array([
    [5, 5]
])

# Goal position (where drone should reach)
GOAL_POSITION = np.array([35, 35])

# Collision parameters
GOAL_REWARD = 100
BATTERY_EMPTY_PENALTY = -100
COLLISION_PENALTY = -100
PROGRESS_REWARD_MULTIPLIER = 2
PROGRESS_REWARD_EPLISON = 0.001
STEP_PENALTY = -1

# Threshold parameters
GOAL_THRESHOLD = 0.25      # Distance threshold to consider goal reached, should not be exactly 0
OBSTACLE_THRESHOLD = 0.25  # Radius around obstacle for collision, should not be exactly 0

# MDP parameters
DISCOUNT_FACTOR = 0.95    # Discount factor for future rewards

# Plotting parameters
PLOT_FIGURE_SIZE = 1500