import math
from dataclasses import dataclass
from typing import List, Tuple

# --- GRID CONSTANTS ---
GRID_WIDTH = 40
GRID_HEIGHT = 40

# --- SENSOR CONSTANTS ---
SENSOR_ACCURACY = 0.8
SENSOR_RANGE = 2

# --- LOG-ODDS CONSTANTS ---
# L = log(p / (1-p))
# If p=0.8, L = log(4) ~= 1.386
LO_OCCUPIED = math.log(SENSOR_ACCURACY / (1.0 - SENSOR_ACCURACY))
LO_FREE = math.log((1.0 - SENSOR_ACCURACY) / SENSOR_ACCURACY)
# Cap the confidence to prevent floating point issues (e.g., -100 to 100)
MAX_CONFIDENCE = 10.0

# --- DIRECTIONS (N, S, E, W, NE, NW, SE, SW, Stay) ---
DIRECTIONS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)
]

NUM_BELIEF_PARTICLES = 500


# --- TEST CASE DEFINITION ---
@dataclass
class TestCase:
    """A test case configuration for the POMDP drone simulation."""
    name: str
    difficulty: str  # "easy", "medium", "hard"
    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
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


# =============================================================================
# TEST CASES
# =============================================================================

# --- EASY TEST CASES ---
# Few obstacles, relatively clear path to goal

EASY_OPEN_FIELD = TestCase(
    name="easy_open_field",
    difficulty="easy",
    start_pos=(0, 0),
    goal_pos=(39, 39),
    obstacles=[
        (10, 10), (10, 11), (10, 12),
        (25, 20), (26, 20), (27, 20),
    ],
    description="Open field with minimal scattered obstacles"
)

EASY_CORRIDOR = TestCase(
    name="easy_corridor",
    difficulty="easy",
    start_pos=(0, 20),
    goal_pos=(39, 20),
    obstacles=(
        _generate_line(5, 15, 5, 19) +  # Small wall below path
        _generate_line(30, 21, 30, 25)  # Small wall above path
    ),
    description="Simple corridor with small walls to navigate around"
)

EASY_DIAGONAL = TestCase(
    name="easy_diagonal",
    difficulty="easy",
    start_pos=(5, 5),
    goal_pos=(35, 35),
    obstacles=[
        (15, 15), (16, 16), (17, 17),  # Small diagonal cluster
        (25, 25), (26, 25), (25, 26),  # Small corner cluster
    ],
    description="Diagonal path with small obstacle clusters"
)

# --- MEDIUM TEST CASES ---
# More obstacles, requires some navigation

MEDIUM_WALL_GAP = TestCase(
    name="medium_wall_gap",
    difficulty="medium",
    start_pos=(0, 20),
    goal_pos=(39, 20),
    obstacles=(
        _generate_line(15, 10, 15, 18) +  # Wall with gap at y=19-21
        _generate_line(15, 22, 15, 30) +
        _generate_line(28, 10, 28, 19) +  # Another wall with gap at y=20-21
        _generate_line(28, 22, 28, 30)
    ),
    description="Two walls with gaps that must be navigated through"
)

MEDIUM_ZIGZAG = TestCase(
    name="medium_zigzag",
    difficulty="medium",
    start_pos=(0, 0),
    goal_pos=(39, 39),
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

HARD_C_SHAPE_TRAP = TestCase(
    name="hard_c_shape_trap",
    difficulty="hard",
    start_pos=(20, 15),  # Inside the C at the bottom
    goal_pos=(20, 35),   # Above the C
    obstacles=_generate_c_shape(20, 20, 16, 12, opening="top"),
    description="Drone starts inside C-shape, must exit and reach goal above"
)

HARD_C_SHAPE_NAVIGATE = TestCase(
    name="hard_c_shape_navigate", 
    difficulty="hard",
    start_pos=(20, 10),  # Below the C-shape
    goal_pos=(8, 20),    # Inside the C on the left side
    obstacles=_generate_c_shape(20, 20, 20, 14, opening="left"),
    description="Drone must navigate around C-shape to enter from the opening"
)

HARD_DOUBLE_C = TestCase(
    name="hard_double_c",
    difficulty="hard",
    start_pos=(5, 5),
    goal_pos=(35, 35),
    obstacles=(
        _generate_c_shape(15, 15, 12, 10, opening="right") +
        _generate_c_shape(28, 28, 12, 10, opening="left")
    ),
    description="Two C-shapes creating a challenging maze-like path"
)

HARD_MAZE_CORRIDOR = TestCase(
    name="hard_maze_corridor",
    difficulty="hard",
    start_pos=(0, 0),
    goal_pos=(39, 39),
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
    goal_pos=(35, 20),   # Far right, outside the C
    obstacles=_generate_c_shape(20, 20, 24, 16, opening="right"),
    description="Drone trapped at bottom of large C-shape, goal is outside on the right"
)


# =============================================================================
# TEST CASE REGISTRY
# =============================================================================

TEST_CASES = {
    # Easy
    "easy_open_field": EASY_OPEN_FIELD,
    "easy_corridor": EASY_CORRIDOR,
    "easy_diagonal": EASY_DIAGONAL,
    # Medium
    "medium_wall_gap": MEDIUM_WALL_GAP,
    "medium_zigzag": MEDIUM_ZIGZAG,
    "medium_scattered": MEDIUM_SCATTERED,
    # Hard
    "hard_c_shape_trap": HARD_C_SHAPE_TRAP,
    "hard_c_shape_navigate": HARD_C_SHAPE_NAVIGATE,
    "hard_double_c": HARD_DOUBLE_C,
    "hard_maze_corridor": HARD_MAZE_CORRIDOR,
    "hard_c_inside_out": HARD_C_INSIDE_OUT,
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
