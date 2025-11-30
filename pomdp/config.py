import math

# --- GRID CONSTANTS ---
GRID_WIDTH = 10
GRID_HEIGHT = 10

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

# --- DIRECTIONS (N, S, E, W, NE, NW, SE, SW) ---
DIRECTIONS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)
]

NUM_BELIEF_PARTICLES = 500
