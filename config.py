import math
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple

seed = random.randint(0, 1000)
rng = np.random.RandomState(seed)

# =============================================================================
# MDP SIMLUATION CONFIGURATION
# =============================================================================

# Grid world parameters
GRID_SIZE = 40            # Zero-indexed, non-inclusive
MAXIMUM_TIME_STEPS = 100  # Zero-indexed, non-inclusive, no more than 100 (otherwise edit wind array)

# Battery parameters
BATTERY_MIN = 0          # Inclusive
BATTERY_MAX = 95         # Inclusive, no more than MAXIMUM_TIME_STEPS
BATTERY_DRAIN_RATE = 1   # Per time step

# Wind parameters (units/s)
# Inclusive
WIND_ENABLE = False
WIND_MIN = -1
WIND_MAX = 1
WIND = np.array([[rng.randint(WIND_MIN, WIND_MAX + 1), rng.randint(WIND_MIN, WIND_MAX + 1)] for _ in range(MAXIMUM_TIME_STEPS)])

# Velocity, Acceleration bounds (units/s) and (units/s^2)
# Inclusive
VELOCITY_MIN = -5
VELOCITY_MAX = 5
ACCEL_MIN = -2
ACCEL_MAX = 2

# =============================================================================
# MDP SINGLE AGENT CONFIGURATION
# =============================================================================

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
ASTAR_MAX_DEVIATION = 3             # Maximum allowed distance from A* path before penalty applies
ASTAR_DEVIATION_PENALTY = -10       # Base Penalty for distances beyond max deviation

# MDP parameters  
DISCOUNT_FACTOR = 0.8
ROLLOUT_NUM_ROLLOUTS = 1
ROLLOUT_MAX_DEPTH = 5

# =============================================================================
# MDP MULTI-AGENT CONFIGURATION
# =============================================================================

# Number of agents in the system, this is overwritten by the configure() function
NUM_AGENTS = 2

# Multi-agent parameters
AGENT_COLLISION_THRESHOLD = 0.5     # Distance threshold for agent-agent collision, should not be exactly 0
AGENT_COLLISION_PENALTY = -10000      # Penalty for colliding with another agent, should not be exactly 0

AGENT_REWARDS = [GOAL_REWARD] * 1000

# =============================================================================
# POMDP CONFIGURATION
# =============================================================================

# POMDP world parameters
GRID_WIDTH = GRID_SIZE
GRID_HEIGHT = GRID_SIZE

# Sensor parameters
SENSOR_ACCURACY = 0.8
SENSOR_RANGE = 2

# Log-odds constants
# L = log(p / (1-p))
# If p=0.8, L = log(4) ~= 1.386
LO_OCCUPIED = math.log(SENSOR_ACCURACY / (1.0 - SENSOR_ACCURACY))
LO_FREE = math.log((1.0 - SENSOR_ACCURACY) / SENSOR_ACCURACY)
# Cap the confidence to prevent floating point issues (e.g., -100 to 100)
MAX_CONFIDENCE = 10.0

# Directions: (N, S, E, W, NE, NW, SE, SW, Stay)
DIRECTIONS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)
]

NUM_BELIEF_PARTICLES = 500

# =============================================================================
# MISC CONFIGURATION
# =============================================================================

# Plotting parameters
LIVE_UPDATE = True
PLOT_FIGURE_SIZE = 1500

# =============================================================================
# LEGACY TEST CASES CONFIGURATION
# =============================================================================

# There will all be overwritten by the configure() function in run_mdp_single_agent.py and run_mdp_multi_agent.py.
# This is left here to be backwards compatible with existing code while using new testing framework.

# Obstacle positions (x, y)
OBSTACLES = rng.randint(0, GRID_SIZE, size=(40, 2))
MAX_OBSTACLES = OBSTACLES.shape[0] # Zero-Indexed, Non-inclusive, See Above

# Drone starting position
STARTING_POSITION = np.array([
    [5, 5]
])

# Goal position
GOAL_POSITION = np.array([35, 35])                

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

# =============================================================================
# TEST CASE DEFINITION
# =============================================================================

@dataclass
class TestCase:
    name: str
    difficulty: str
    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    multi_agent_start_pos: List[Tuple[int, int]]
    multi_agent_goal_pos: List[Tuple[int, int]]
    obstacles: List[Tuple[int, int]]
    description: str = ""

def _generate_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    """Generate a line of obstacle cells from (x1,y1) to (x2,y2)."""
    obstacles = []
    if x1 == x2:  # Vertical line
        for y in range(min(y1, y2), max(y1, y2) + 1):
            obstacles.append((x1, y))
    elif y1 == y2:  # Horizontal line
        for x in range(min(x1, x2), max(x1, x2) + 1):
            obstacles.append((x, y1))
    return obstacles

def _generate_c_shape(center_x: int, center_y: int, width: int, height: int, 
                      opening: str = "right") -> List[Tuple[int, int]]:
    """
    Generate a C-shaped obstacle pattern.
    The opening is where the C is open (right, left, top, bottom).
    """
    obstacles = []
    half_w = width // 2
    half_h = height // 2
    
    if opening == "right":
        # Left vertical wall
        obstacles.extend(_generate_line(center_x - half_w, center_y - half_h, 
                                        center_x - half_w, center_y + half_h))
        # Top horizontal wall
        obstacles.extend(_generate_line(center_x - half_w, center_y + half_h, 
                                        center_x + half_w, center_y + half_h))
        # Bottom horizontal wall
        obstacles.extend(_generate_line(center_x - half_w, center_y - half_h, 
                                        center_x + half_w, center_y - half_h))
    elif opening == "left":
        # Right vertical wall
        obstacles.extend(_generate_line(center_x + half_w, center_y - half_h, 
                                        center_x + half_w, center_y + half_h))
        # Top horizontal wall
        obstacles.extend(_generate_line(center_x - half_w, center_y + half_h, 
                                        center_x + half_w, center_y + half_h))
        # Bottom horizontal wall
        obstacles.extend(_generate_line(center_x - half_w, center_y - half_h, 
                                        center_x + half_w, center_y - half_h))
    elif opening == "top":
        # Bottom horizontal wall
        obstacles.extend(_generate_line(center_x - half_w, center_y - half_h, 
                                        center_x + half_w, center_y - half_h))
        # Left vertical wall
        obstacles.extend(_generate_line(center_x - half_w, center_y - half_h, 
                                        center_x - half_w, center_y + half_h))
        # Right vertical wall
        obstacles.extend(_generate_line(center_x + half_w, center_y - half_h, 
                                        center_x + half_w, center_y + half_h))
    elif opening == "bottom":
        # Top horizontal wall
        obstacles.extend(_generate_line(center_x - half_w, center_y + half_h, 
                                        center_x + half_w, center_y + half_h))
        # Left vertical wall
        obstacles.extend(_generate_line(center_x - half_w, center_y - half_h, 
                                        center_x - half_w, center_y + half_h))
        # Right vertical wall
        obstacles.extend(_generate_line(center_x + half_w, center_y - half_h, 
                                        center_x + half_w, center_y + half_h))
    
    return list(set(obstacles))  # Remove duplicates

def _generate_circle(center_x: int, center_y: int, radius: int) -> List[Tuple[int, int]]:
    """
    Generate a circular obstacle pattern using Bresenham's circle algorithm.
    Returns points forming a circle outline.
    """
    obstacles = []
    x = 0
    y = radius
    d = 3 - 2 * radius
    
    def add_circle_points(cx, cy, x, y):
        """Add all 8 symmetric points of the circle"""
        points = [
            (cx + x, cy + y), (cx - x, cy + y),
            (cx + x, cy - y), (cx - x, cy - y),
            (cx + y, cy + x), (cx - y, cy + x),
            (cx + y, cy - x), (cx - y, cy - x)
        ]
        return points
    
    while x <= y:
        obstacles.extend(add_circle_points(center_x, center_y, x, y))
        if d < 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y -= 1
        x += 1
    
    return list(set(obstacles))  # Remove duplicates


# =============================================================================
# TEST CASES
# =============================================================================

# --- EASY TEST CASES ---
# Few obstacles, relatively clear path to goal

EASY_EMPTY_FIELD = TestCase(
    name="easy_empty_field",
    difficulty="easy",
    start_pos=(37, 37),
    goal_pos=(3, 3),
    multi_agent_start_pos=[(1,5),(5,1),(35,35)],
    multi_agent_goal_pos=[(35,30),(30,35),(5,5)],
    obstacles=[],
    description="Empty field with no obstacles"
)

EASY_SCATTERED_OBSTACLES = TestCase(
    name="easy_scattered_obstacles",
    difficulty="easy",
    start_pos=(5, 5),
    goal_pos=(35, 35),
    multi_agent_start_pos=[(5, 5)],
    multi_agent_goal_pos=[(35, 35)],
    obstacles = [
        (1, 15), (4, 9), (8, 12), (10, 23), (12, 26), (13, 5), (15, 11), (16, 17), (18, 14), (19, 26),
        (20, 16), (22, 6), (23, 17), (24, 8), (25, 34), (28, 19), (29, 12), (30, 4), (31, 28), (33, 24),
        (12, 33), (17, 27), (21, 2), (27, 31), (5, 20), (7, 23), (6, 28), (38, 12), (36, 25), (14, 28),
        (11, 18), (9, 33), (28, 21), (32, 11), (16, 8), (18, 36), (25, 7), (35, 2), (22, 15), (38, 29)
    ],
    description="Open field with random scattered obstacles"
)

EASY_OPEN_FIELD = TestCase(
    name="easy_open_field",
    difficulty="easy",
    start_pos=(0, 0),
    goal_pos=(39, 39),
    multi_agent_start_pos=[(0, 0)],
    multi_agent_goal_pos=[(39, 39)],
    obstacles=[
        (10, 10), (10, 11), (10, 12),
        (25, 20), (26, 20), (27, 20),
    ],
    description="Open field with minimal clustered obstacles"
)

EASY_CORRIDOR = TestCase(
    name="easy_corridor",
    difficulty="easy",
    start_pos=(0, 20),
    goal_pos=(39, 20),
    multi_agent_start_pos=[(0, 20)],
    multi_agent_goal_pos=[(39, 20)],
    obstacles=(
        _generate_line(5, 15, 5, 19) +  # Small wall below path
        _generate_line(30, 21, 30, 25)  # Small wall above path
    ),
    description="Simple corridor with small walls to navigate around"
)

# --- MEDIUM TEST CASES ---
# More obstacles, requires some navigation

MEDIUM_WALL_GAP = TestCase(
    name="medium_wall_gap",
    difficulty="medium",
    start_pos=(0, 20),
    goal_pos=(39, 20),
    multi_agent_start_pos=[(0, 20)],
    multi_agent_goal_pos=[(39, 20)],
    obstacles=(
        _generate_line(15, 10, 15, 17) +  # Wall with gap at y=19-21
        _generate_line(15, 23, 15, 30) +
        _generate_line(28, 10, 28, 17) +  # Another wall with gap at y=20-21
        _generate_line(28, 23, 28, 30)
    ),
    description="Two walls with gaps that must be navigated through"
)

MEDIUM_ZIGZAG = TestCase(
    name="medium_zigzag",
    difficulty="medium",
    start_pos=(0, 0),
    goal_pos=(39, 39),
    multi_agent_start_pos=[(0, 0)],
    multi_agent_goal_pos=[(39, 39)],
    obstacles=(
        _generate_line(10, 0, 10, 25) +   # First vertical wall
        _generate_line(20, 15, 20, 39) +  # Second vertical wall
        _generate_line(30, 0, 30, 25)     # Third vertical wall
    ),
    description="Zigzag path required through vertical walls"
)

MEDIUM_SCATTERED = TestCase(
    name="medium_scattered",
    difficulty="medium",
    start_pos=(2, 2),
    goal_pos=(37, 37),
    multi_agent_start_pos=[(2, 2)],
    multi_agent_goal_pos=[(37, 37)],
    obstacles=[
        # Multiple small clusters scattered across the grid
        (8, 8), (8, 9), (9, 8), (9, 9),
        (15, 5), (16, 5), (15, 6), (16, 6),
        (5, 15), (5, 16), (6, 15), (6, 16),
        (20, 20), (20, 21), (21, 20), (21, 21), (22, 20), (22, 21),
        (30, 10), (31, 10), (30, 11), (31, 11),
        (12, 28), (13, 28), (12, 29), (13, 29),
        (28, 25), (29, 25), (28, 26), (29, 26),
        (35, 30), (35, 31), (36, 30), (36, 31),
    ],
    description="Multiple scattered obstacle clusters requiring careful navigation"
)

# --- HARD TEST CASES ---
# Complex obstacle patterns, require significant planning

HARD_MAZE_CORRIDOR = TestCase(
    name="hard_maze_corridor",
    difficulty="hard",
    start_pos=(1, 1),
    goal_pos=(7, 27),
    multi_agent_start_pos=[(1, 1)],
    multi_agent_goal_pos=[(7, 27)],
    obstacles=(
        # Create a maze-like pattern
        _generate_line(5, 0, 5, 30) +
        _generate_line(5, 30, 15, 30) +
        _generate_line(15, 10, 15, 30) +
        _generate_line(15, 10, 25, 10) +
        _generate_line(25, 10, 25, 35) +
        _generate_line(25, 35, 35, 35) +
        _generate_line(35, 20, 35, 35)
    ),
    description="Complex maze requiring multiple direction changes"
)

HARD_C_INSIDE_OUT = TestCase(
    name="hard_c_inside_out",
    difficulty="hard",
    start_pos=(20, 14),  # Inside the C at the very bottom
    goal_pos=(2, 20),    # Far left, outside the C
    multi_agent_start_pos=[(20, 14)],
    multi_agent_goal_pos=[(2, 20)],
    obstacles=_generate_c_shape(20, 20, 24, 16, opening="right"),
    description="Drone trapped at bottom of large C-shape, goal is outside on the right"
)

HARD_TWO_LINE_MAZE = TestCase(
    name="hard_two_line_maze",
    difficulty="hard",
    start_pos=(5, 5),
    goal_pos=(35, 35),
    multi_agent_start_pos=[(5, 5),(5, 3),(5, 1)],
    multi_agent_goal_pos=[(35, 35),(35, 33),(35, 31)],
    obstacles=(
        _generate_line(0, 12, 24, 12) +   # First horizontal line at y=12
        _generate_line(15, 25, 39, 25)    # Second horizontal line at y=25
    ),
    description="Two staggered horizontal walls creating a maze-like navigation challenge"
)

HARD_SPIRAL_MAZE = TestCase(
    name="hard_spiral_maze",
    difficulty="hard",
    start_pos=(2, 2),
    goal_pos=(17, 20),
    multi_agent_start_pos=[(2, 2)],
    multi_agent_goal_pos=[(17, 20)],
    obstacles=(
        # Outer boundary walls
        _generate_line(0, 0, 39, 0) +      # Bottom wall
        _generate_line(0, 0, 0, 39) +      # Left wall
        _generate_line(0, 39, 39, 39) +    # Top wall
        _generate_line(39, 0, 39, 39) +    # Right wall
        # First inner spiral layer
        _generate_line(0, 5, 34, 5) +      # Bottom
        _generate_line(34, 5, 34, 34) +    # Right
        _generate_line(5, 34, 34, 34) +    # Top
        _generate_line(5, 10, 5, 34) +     # Left
        # Second inner spiral layer
        _generate_line(5, 10, 29, 10) +    # Bottom
        _generate_line(29, 10, 29, 29) +   # Right
        _generate_line(11, 29, 29, 29) +   # Top
        _generate_line(10, 15, 10, 29) +   # Left
        # Third inner spiral layer
        _generate_line(10, 15, 24, 15) +   # Bottom
        _generate_line(24, 15, 24, 24) +   # Right
        _generate_line(15, 24, 24, 24) +   # Top
        _generate_line(15, 20, 15, 24)     # Left
    ),
    description="Complex spiral maze with multiple nested layers requiring careful navigation to the center"
)

HARD_HAIRPIN_TURN = TestCase(
    name="hard_hairpin_turn",
    difficulty="hard",
    start_pos=(18, 3),   # Bottom middle, slightly left
    goal_pos=(22, 3),    # Bottom middle, slightly right
    multi_agent_start_pos=[(18, 3)],
    multi_agent_goal_pos=[(22, 3)],
    obstacles=(
        # Left corridor outer wall (stops before roundabout, one unit wider)
        _generate_line(11, 0, 11, 26) +
        # Middle divider wall (separates up and down corridors, stops before roundabout)
        _generate_line(20, 0, 20, 26) +
        # Right corridor outer wall (stops before roundabout, one unit wider)
        _generate_line(29, 0, 29, 26) +
        # Roundabout at the top (smaller and lower)
        _generate_circle(20, 28, 3) +     # Central roundabout circle
        _generate_circle(20, 28, 2) +     # Central roundabout circle
        # Connect corridors to roundabout
        _generate_line(11, 27, 11, 30) +  # Left wall extension to roundabout
        _generate_line(29, 27, 29, 30) +  # Right wall extension to roundabout
        # Top boundary above roundabout
        _generate_line(11, 35, 29, 35) +
        _generate_line(11, 30, 11, 35) +  # Left boundary
        _generate_line(29, 30, 29, 35) +  # Right boundary
        # Bottom barriers to force correct entry/exit
        _generate_line(11, 0, 19, 0) +    # Block left side except for start entrance
        _generate_line(21, 0, 29, 0)      # Block right side except for goal exit
    ),
    description="Racetrack-style hairpin with roundabout: navigate up the left corridor, around the circular roundabout at top, and down the right corridor"
)


# =============================================================================
# TEST CASE REGISTRY
# =============================================================================

TEST_CASES = {
    # Easy
    "easy_empty_field": EASY_EMPTY_FIELD,
    "easy_scattered_obstacles": EASY_SCATTERED_OBSTACLES,
    "easy_open_field": EASY_OPEN_FIELD,
    "easy_corridor": EASY_CORRIDOR,
    # Medium
    "medium_wall_gap": MEDIUM_WALL_GAP,
    "medium_zigzag": MEDIUM_ZIGZAG,
    "medium_scattered": MEDIUM_SCATTERED,
    # Hard
    "hard_maze_corridor": HARD_MAZE_CORRIDOR,
    "hard_c_inside_out": HARD_C_INSIDE_OUT,
    "hard_two_line_maze": HARD_TWO_LINE_MAZE,
    "hard_spiral_maze": HARD_SPIRAL_MAZE,
    "hard_hairpin_turn": HARD_HAIRPIN_TURN,
}


def get_test_case(name: str) -> TestCase:
    """Get a test case by name."""
    if name not in TEST_CASES:
        available = ", ".join(TEST_CASES.keys())
        raise ValueError(f"Unknown test case '{name}'. Available: {available}")
    return TEST_CASES[name]


def list_test_cases(difficulty: str = None) -> List[str]:
    """List all test cases, optionally filtered by difficulty."""
    if difficulty is None:
        return list(TEST_CASES.keys())
    return [name for name, tc in TEST_CASES.items() if tc.difficulty == difficulty]
