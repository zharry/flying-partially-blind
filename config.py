import numpy as np
import random

seed = random.randint(0, 1000)
rng = np.random.RandomState(seed)

# Grid world parameters
GRID_SIZE = 40           # Zero-indexed, non-inclusive
MAXIMUM_TIME_STEPS = 100  # Zero-indexed, non-inclusive, no more than 100 (otherwise edit wind array)

# Velocity bounds (units/s)
VELOCITY_MIN = -10        # Inclusive
VELOCITY_MAX = 10         # Inclusive

# Battery parameters
BATTERY_MIN = 0          # Inclusive
BATTERY_MAX = 95        # Inclusive, no more than MAXIMUM_TIME_STEPS
BATTERY_DRAIN_RATE = 1   # Per time step

# Wind parameters (units/s)
WIND_MIN = -1            # Inclusive, See Below
WIND_MAX = 1             # Inclusive, See Below
# WIND = np.array([[0, 0] for _ in range(MAXIMUM_TIME_STEPS)])
WIND = np.array([[rng.randint(WIND_MIN, WIND_MAX), rng.randint(WIND_MIN, WIND_MAX)] for _ in range(MAXIMUM_TIME_STEPS)])

# Action bounds (acceleration, units/s^2)
ACCEL_MIN = -2           # Inclusive
ACCEL_MAX = 2            # Inclusive

# Obstacle positions (x, y)
OBSTACLES = rng.randint(0, GRID_SIZE, size=(40, 2))
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
COLLISION_PENALTY = -10000
OUT_OF_BOUNDS_PENALTY = -1000
PROGRESS_REWARD_MULTIPLIER = 100
PROGRESS_REWARD_EPSILON = 0.001
STEP_PENALTY = 0

# Threshold parameters
GOAL_THRESHOLD = 0.25      # Distance threshold to consider goal reached, should not be exactly 0
OBSTACLE_THRESHOLD = 1.0  # Radius around obstacle for collision, should not be exactly 0

# A* parameters
ASTAR_LOOKAHEAD_DISTANCE = 5        # Number of waypoints to look ahead
ASTAR_OBSTACLE_CLEARANCE = 2        # Minimum distance to maintain from obstacles when planning A* path
ASTAR_MAX_DEVIATION = 5             # Maximum allowed distance from A* path before penalty applies
ASTAR_DEVIATION_PENALTY = -20.0     # Base Penalty for distances beyond max deviation

# MDP parameters  
DISCOUNT_FACTOR = 0.8
ROLLOUT_NUM_ROLLOUTS = 1
ROLLOUT_MAX_DEPTH = 15

# Multi-agent parameters
NUM_AGENTS = 2                       # Number of agents in the system
AGENT_COLLISION_THRESHOLD = 0.25     # Distance threshold for agent-agent collision, should not be exactly 0
AGENT_COLLISION_PENALTY = -10000      # Penalty for colliding with another agent, should not be exactly 0

# Multi-agent goal positions, must have length NUM_AGENTS
MULTI_AGENT_GOAL_POSITIONS = np.array([
    [20, 35],
    [35, 20],
])

# Multi-agent starting positions, must have length NUM_AGENTS
MULTI_AGENT_STARTING_POSITIONS = np.array([
    [10, 5],
    [5, 10],
])

# Plotting parameters
LIVE_UPDATE = True
PLOT_FIGURE_SIZE = 1500