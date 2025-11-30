# Flying Partially Blind

Final Project for Stanford AA228: Decision Making Under Uncertainty

## Project Overview

This project implements a Markov Decision Process (MDP) for autonomous drone navigation in a 2D grid world with dynamic wind conditions, obstacles, and battery constraints. The drone must navigate from a starting position to a goal position while avoiding obstacles and managing its battery consumption under uncertain wind conditions.

## Problem Formulation

### MDP Components

The drone navigation problem is formulated as a finite-horizon MDP with the following components:

- **State Space (S)**: Continuous state space discretized into a grid
- **Action Space (A)**: Discrete 2D acceleration commands
- **Transition Function (T)**: Deterministic physics-based transitions with collision detection
- **Reward Function (R)**: Sparse rewards with progress incentives
- **Discount Factor (γ)**: 0.8
- **Time Horizon**: Maximum of 100 time steps

## State Space

The state `s ∈ S` is represented as a 7-dimensional vector:

```
s = [x, y, vx, vy, battery, wx, wy]
```

### State Components

1. **Position (x, y)**
   - Drone's location in 2D grid
   - Range: [0, 39] × [0, 39] (40×40 grid)
   - Type: Integer coordinates
   - Clipped to grid boundaries

2. **Velocity (vx, vy)**
   - Drone's velocity in x and y directions
   - Range: [-3, 3] × [-3, 3] units/second
   - Type: Integer values
   - Affected by acceleration and wind

3. **Battery Level**
   - Remaining battery charge
   - Range: [0, 95] units
   - Initial value: 95 units
   - Drains at 1 unit per time step

4. **Wind (wx, wy)**
   - Current wind velocity affecting drone motion
   - Range: [-1, 1] × [-1, 1] units/second
   - Type: Integer values
   - Time-varying (pre-defined wind pattern)
   - Directly affects velocity updates

### State Space Size

Total number of discrete states:
```
|S| = 40 × 40 × 7 × 7 × 96 × 3 × 3 = 20,321,280 states
```

### Terminal States

A state is terminal if any of the following conditions are met:

1. **Goal Reached**: `||position - goal|| ≤ 0.25`
2. **Collision**: `||position - obstacle|| ≤ 0.50` for any obstacle
3. **Battery Depleted**: `battery ≤ 0`

## Action Space

The action `a ∈ A` represents 2D acceleration:

```
a = [ax, ay]
```

### Action Components

- **Acceleration (ax, ay)**
  - Drone's acceleration command in x and y directions
  - Range: [-1, 1] × [-1, 1] units/second²
  - Type: Integer values
  - Total actions: 3 × 3 = 9 discrete actions

### Available Actions

The drone can choose from 9 discrete acceleration commands:
```
{(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)}
```

## Transition Function

The transition function `T(s'|s,a)` is deterministic and physics-based:

```
s' = T(s, a)
```

### Transition Dynamics

Given current state `s = [x, y, vx, vy, b, wx, wy]` and action `a = [ax, ay]`:

1. **Velocity Update**:
   ```
   v'x = clip(vx + ax + wx, -3, 3)
   v'y = clip(vy + ay + wy, -3, 3)
   ```
   - New velocity = current velocity + acceleration + wind
   - Clipped to velocity bounds [-3, 3]

2. **Position Update**:
   ```
   x' = clip(x + v'x, 0, 39)
   y' = clip(y + v'y, 0, 39)
   ```
   - New position = current position + new velocity
   - Clipped to grid boundaries
   - **Collision Detection**: If path intersects obstacle (within radius 0.5), drone stops at obstacle position

3. **Battery Update**:
   ```
   b' = clip(b - 1, 0, 95)
   ```
   - Battery drains by 1 unit per time step
   - Minimum battery level is 0

4. **Wind Update**:
   ```
   w' = WIND[t+1]
   ```
   - Wind follows pre-defined time-varying pattern
   - Updated based on next time step

### Collision Detection

The transition function includes sophisticated collision detection:

- **Path-based collision**: Checks if the line segment from current to next position intersects any obstacle
- **Obstacle radius**: 0.5 units around each obstacle center
- **Collision behavior**: Drone stops at the obstacle position if collision detected

## Reward Function

The reward function `R(s, a, s')` provides feedback for the agent's actions:

### Reward Components

1. **Goal Reward** (Terminal):
   ```
   R = +10,000 if ||s'.position - goal|| ≤ 0.25
   ```
   - Large positive reward for reaching the goal

2. **Collision Penalty** (Terminal):
   ```
   R = -1,000 if collision with any obstacle
   ```
   - Large negative penalty for crashing

3. **Battery Depletion Penalty** (Terminal):
   ```
   R = -100 if battery ≤ 0
   ```
   - Moderate penalty for running out of battery

4. **Out of Bounds Penalty** (Non-terminal):
   ```
   R = -1,000 if touching grid boundary
   ```
   - Large penalty for reaching grid edges

5. **Progress Reward** (Non-terminal):
   ```
   R = 100 / (0.001 + ||s'.position - goal||)
   ```
   - Encourages movement toward goal
   - Inversely proportional to distance to goal
   - Higher reward when closer to goal

6. **Step Penalty** (Non-terminal):
   ```
   R = 0 (configurable)
   ```
   - Optional constant penalty per time step
   - Currently set to 0

### Total Reward

For non-terminal states:
```
R(s, a, s') = STEP_PENALTY + PROGRESS_REWARD
```

For terminal states:
```
R(s, a, s') = GOAL_REWARD | COLLISION_PENALTY | BATTERY_PENALTY
```

## Environment Configuration

### Grid World Setup

- **Grid Size**: 40 × 40
- **Starting Position**: [5, 5]
- **Goal Position**: [35, 35]
- **Number of Obstacles**: 40 randomly placed obstacles
- **Goal Threshold**: 0.25 units (Euclidean distance)
- **Obstacle Threshold**: 0.50 units (collision radius)

### Physical Parameters

- **Max Time Steps**: 100
- **Battery Capacity**: 95 units
- **Battery Drain**: 1 unit/step
- **Velocity Range**: [-3, 3] units/s
- **Acceleration Range**: [-1, 1] units/s²
- **Wind Range**: [-1, 1] units/s

## Solution Approach

### Policy

The project implements a **Rollout Policy** with the following characteristics:

1. **Rollout Planner**:
   - Evaluates all 9 possible actions
   - Performs multiple rollouts (15) for each action
   - Simulates future states up to depth 10
   - Selects action with highest expected return

2. **Base Policy** (RandomGreedyPolicy):
   - 33% probability: Greedy action (move toward goal)
   - 67% probability: Random exploration
   - Used during rollout simulations

3. **Greedy Heuristic**:
   - Computes direction to goal
   - Accounts for current velocity and wind
   - Calculates desired acceleration toward goal

### Value Estimation

The rollout planner estimates action values using:

```
Q(s, a) ≈ (1/N) Σ [R(s,a,s') + γ R(s',π(s'),s'') + ... ]
```

Where:
- N = 15 rollouts per action
- γ = 0.8 discount factor
- Maximum rollout depth = 10 steps

## Implementation Details

### Project Structure

```
flying-partially-blind/
├── config.py                 # Configuration parameters
├── drone/
│   ├── action.py            # DroneAction class
│   └── state.py             # DroneState class
├── mdp/
│   ├── drone.py             # DroneMDP (transition & reward)
│   ├── simulator.py         # Episode simulation
│   └── visualization.py     # Plotting utilities
├── policy/
│   └── rollout.py           # Rollout planner & base policies
├── main.py                  # Main execution script
└── requirements.txt         # Python dependencies
```

### Key Classes

1. **DroneState**: Encapsulates state representation with validation
2. **DroneAction**: Represents discrete acceleration actions
3. **DroneMDP**: Implements transition and reward functions
4. **Simulator**: Manages episode execution and history tracking
5. **RolloutPlanner**: Implements rollout-based decision making

## Running the Simulation

### Installation

```bash
pip install -r requirements.txt
```

### Execution

```bash
python main.py
```

The simulation will:
1. Initialize the drone at starting position
2. Execute rollout planning at each time step
3. Display live visualization of drone trajectory
4. Print state, action, and reward information
5. Show final trajectory and total reward

### Visualization

The visualization shows:
- Grid world with obstacles (red circles)
- Goal position (yellow star)
- Drone trajectory (blue line with green markers)
- Current drone position (large green circle)
- Wind vectors (optional)

## Key Features

1. **Realistic Physics**: Velocity and position updates based on Newtonian mechanics
2. **Dynamic Wind**: Time-varying wind affects drone motion
3. **Battery Constraint**: Finite battery creates urgency
4. **Sophisticated Collision Detection**: Path-based obstacle avoidance
5. **Sparse Rewards**: Primarily goal-directed with progress shaping
6. **Rollout Planning**: Monte Carlo tree search approximation
7. **Live Visualization**: Real-time trajectory plotting

## Challenges and Considerations

1. **Large State Space**: ~20 million discrete states
2. **Partial Observability**: Wind changes are pre-determined but must be observed
3. **Sparse Rewards**: Goal reward requires navigating through long trajectories
4. **Exploration-Exploitation**: Balance between greedy goal-seeking and exploration
5. **Computational Complexity**: Rollout planning requires many forward simulations

## Future Enhancements

- Value function approximation for faster planning
- Online learning of wind patterns
- Multi-drone coordination
- Continuous state/action spaces with function approximation
- Probabilistic wind models for true partial observability

## References

This project was developed for Stanford's AA228: Decision Making Under Uncertainty course.
