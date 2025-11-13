import numpy as np

# Grid world parameters
GRID_SIZE = 40           # Zero-indexed, non-inclusive
MAXIMUM_TIME_STEPS = 1000  # Zero-indexed, non-inclusive, no more than 100 (otherwise edit wind array)

# Velocity bounds (units/s)
VELOCITY_MIN = -3        # Inclusive
VELOCITY_MAX = 3         # Inclusive

# Battery parameters
BATTERY_MIN = 0          # Inclusive
BATTERY_MAX = 995         # Inclusive, no more than MAXIMUM_TIME_STEPS
BATTERY_DRAIN_RATE = 1   # Per time step

# Wind parameters (units/s)
WIND_MIN = -1            # Inclusive, See Below
WIND_MAX = 1             # Inclusive, See Below
WIND = np.array([[0, 0] for _ in range(MAXIMUM_TIME_STEPS)])
# WIND = np.array([
#     [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, -1], [1, -1], [1, -1], [0, -1], [0, -1], [0, -1], [-1, -1], [-1, -1], [-1, 0], [-1, 0], [-1, 0], [-1, 1], [-1, 1], [-1, 1], [0, 1],
#     [0, 1], [0, 0], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, -1], [1, -1], [0, -1], [0, -1], [-1, -1], [-1, -1], [-1, 0], [-1, 0], [-1, 1], [-1, 1], [0, 1], [0, 0], [0, 1], [0, 1], [1, 1], [1, 0], [1, 0],
#     [1, -1], [1, -1], [0, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 0], [1, -1], [1, -1], [0, -1], [0, -1], [-1, -1], [-1, -1], [-1, 0], [-1, 0], [-1, 1],
#     [-1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [1, 0], [1, 0], [1, -1], [1, -1], [0, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, -1], [1, -1], [0, -1], [-1, -1], [-1, 0]
# ])

# Action bounds (acceleration, units/s^2)
ACCEL_MIN = -1           # Inclusive
ACCEL_MAX = 1            # Inclusive

# Obstacle positions (x, y)
# Square Spiral Maze Obstacles (5-unit corridor width)
OBSTACLES = np.array(
    [[x, 0] for x in range(0, 40)] + 
    [[0, y] for y in range(0, 40)] +
    [[x, 39] for x in range(0, 40)] + 
    [[39, y] for y in range(0, 40)] +
    
    [[x, 5] for x in range(0, 35)] +
    [[34, y] for y in range(5, 35)] +
    [[x, 34] for x in range(5, 35)] +
    [[5, y] for y in range(10, 35)] +
    
    [[x, 10] for x in range(5, 30)] +
    [[29, y] for y in range(10, 30)] +
    [[x, 29] for x in range(11, 30)] +
    [[10, y] for y in range(15, 30)] +
    
    [[x, 15] for x in range(10, 25)] +
    [[24, y] for y in range(15, 25)] +
    [[x, 24] for x in range(15, 25)] +
    [[15, y] for y in range(20, 25)]
)
MAX_OBSTACLES = OBSTACLES.shape[0] # Zero-Indexed, Non-inclusive, See Above

# Drone starting position
STARTING_POSITION = np.array([
    [2, 2]
])

# Goal position
GOAL_POSITION = np.array([17, 20])

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
ROLLOUT_NUM_ROLLOUTS = 10
ROLLOUT_MAX_DEPTH = 100

# Plotting parameters
LIVE_UPDATE = True
PLOT_FIGURE_SIZE = 1500