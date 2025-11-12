import sys
sys.path.append("..")

import numpy as np
import config
from drone.action import DroneAction
from drone.state import DroneState

def config_validation_test():
    # Check wind array length
    if len(config.WIND) < config.MAXIMUM_TIME_STEPS:
        raise ValueError(f"Wind array length {len(config.WIND)} is not greater or equal to maximum time steps {config.MAXIMUM_TIME_STEPS}")

    # Check wind values
    for i, wind in enumerate(config.WIND):
        if wind[0] < config.WIND_MIN or wind[0] > config.WIND_MAX:
            raise ValueError(f"Wind[{i}] wx value {wind[0]} is not between {config.WIND_MIN} and {config.WIND_MAX}")
        if wind[1] < config.WIND_MIN or wind[1] > config.WIND_MAX:
            raise ValueError(f"Wind[{i}] wy value {wind[1]} is not between {config.WIND_MIN} and {config.WIND_MAX}")

    # Check obstacle length
    if len(config.OBSTACLES) != config.MAX_OBSTACLES:
        raise ValueError(f"Obstacles array length {len(config.OBSTACLES)} is not equal to maximum obstacles {config.MAX_OBSTACLES}")

    # Check obstacle values
    for i, obstacle in enumerate(config.OBSTACLES):
        if obstacle[0] < 0 or obstacle[0] >= config.GRID_SIZE:
            raise ValueError(f"Obstacle[{i}] x value {obstacle[0]} is not between >= 0 and <{config.GRID_SIZE}")
        if obstacle[1] < 0 or obstacle[1] >= config.GRID_SIZE:
            raise ValueError(f"Obstacle[{i}] y value {obstacle[1]} is not between >= 0 and <{config.GRID_SIZE}")

    # Check starting position
    for i in range(len(config.STARTING_POSITION)):
        if config.STARTING_POSITION[i][0] < 0 or config.STARTING_POSITION[i][0] >= config.GRID_SIZE:
            raise ValueError(f"Starting position[{i}] x value {config.STARTING_POSITION[i][0]} is not between >= 0 and <{config.GRID_SIZE}")
        if config.STARTING_POSITION[i][1] < 0 or config.STARTING_POSITION[i][1] >= config.GRID_SIZE:
            raise ValueError(f"Starting position[{i}] y value {config.STARTING_POSITION[i][1]} is not between >= 0 and <{config.GRID_SIZE}")

        # Check starting position is not an obstacle
        if any(np.array_equal(config.STARTING_POSITION[i], obstacle) for obstacle in config.OBSTACLES):
            raise ValueError(f"Starting position[{i}] {config.STARTING_POSITION[i]} is an obstacle")

    # Check goal position
    if config.GOAL_POSITION[0] < 0 or config.GOAL_POSITION[0] >= config.GRID_SIZE:
        raise ValueError(f"Goal position x value {config.GOAL_POSITION[0]} is not between >= 0 and <{config.GRID_SIZE}")
    if config.GOAL_POSITION[1] < 0 or config.GOAL_POSITION[1] >= config.GRID_SIZE:
        raise ValueError(f"Goal position y value {config.GOAL_POSITION[1]} is not between >= 0 and <{config.GRID_SIZE}")

    # Check goal is not an obstacle
    if any(np.array_equal(config.GOAL_POSITION, obstacle) for obstacle in config.OBSTACLES):
        raise ValueError(f"Goal position {config.GOAL_POSITION} is an obstacle")

    # Check goal threshold
    if config.GOAL_THRESHOLD <= 0:
        raise ValueError(f"Goal threshold {config.GOAL_THRESHOLD} is not > 0")

    # Check obstacle threshold
    if config.OBSTACLE_THRESHOLD <= 0:
        raise ValueError(f"Obstacle threshold {config.OBSTACLE_THRESHOLD} is not > 0")
    
    print("Config validation passed")

def drone_action_test():    
    action = DroneAction(
        acceleration=np.array([0, 0])
    )
    print("DroneAction: __init__() initial test passed", action)

    # Test set_acceleration()
    action.set_acceleration(np.array([1, 1]))
    if not np.all(action.acceleration == np.array([1, 1])):
        raise ValueError("DroneAction: set_acceleration(), get_acceleration() initial test failed", action)
    print("DroneAction: set_acceleration(), get_acceleration() initial test passed")

    # Test set_acceleration() using all possible accelerations
    for ax in range(config.ACCEL_MIN, config.ACCEL_MAX + 1):
        for ay in range(config.ACCEL_MIN, config.ACCEL_MAX + 1):
            action.set_acceleration(np.array([ax, ay]))
            if not np.all(action.acceleration == np.array([ax, ay])):
                raise ValueError("DroneAction: set_acceleration(), get_acceleration() test failed", action)
    print("DroneAction: set_acceleration(), get_acceleration() all test passed")

    # Test from_array(), to_array()
    action_from_array = DroneAction.from_array(action.to_array())
    if not np.all(action_from_array.to_array() == action.to_array()):
        raise ValueError("DroneAction: to_array(), from_array() initialtest failed", action)
    print("DroneAction: to_array(), from_array() initial test passed")

    # Test from_array(), to_array(), from_int(), to_int() using all possible actions
    action_space_size = 0
    last_action = None
    for ax in range(config.ACCEL_MIN, config.ACCEL_MAX + 1):
        for ay in range(config.ACCEL_MIN, config.ACCEL_MAX + 1):
            action_space_size += 1
            action = DroneAction(
                acceleration=np.array([ax, ay])
            )
            last_action = action
            action_array = action.to_array()
            action_int = action.to_int()
            a1 = DroneAction.from_array(action_array).to_array()
            a2 = DroneAction.from_int(action_int).to_array()
            if not np.all(a1 == a1) or not np.all(a1 == action.to_array()):
                raise ValueError("DroneAction: from_array(), to_array(), from_int(), to_int() all test failed", action, a1, a2)
    print("DroneAction: from_array(), to_array(), from_int(), to_int() all test passed")

    # Test get_action_space_size()
    if action_space_size != DroneAction.get_action_space_size():
        raise ValueError("DroneAction: get_action_space_size() test failed", action_space_size, DroneAction.get_action_space_size())
    if action_space_size - 1 != last_action.to_int():
        raise ValueError("DroneAction: get_action_space_size() test failed", last_action.to_int(), action_space_size - 1)
    print("DroneAction: get_action_space_size() test passed: total actions =", action_space_size)

    # Test get_action_space()
    action_space = DroneAction.get_action_space()
    if len(action_space) != (config.ACCEL_MAX - config.ACCEL_MIN + 1) * (config.ACCEL_MAX - config.ACCEL_MIN + 1):
        raise ValueError("DroneAction: get_action_space() test failed", action_space)
    print("DroneAction: get_action_space() test passed")

def drone_state_test():
    drone = DroneState(
        position=config.STARTING_POSITION[0],
        velocity=np.array([0, 0]),
        battery=config.BATTERY_MAX,
        wind=config.WIND[0]
    )
    print("DroneState: __init__() initial test passed", drone)

    # Test from_array() and to_array()
    drone_from_array = DroneState.from_array(drone.to_array())
    if not np.all(drone_from_array.to_array() == drone.to_array()):
        raise ValueError("DroneState: from_array(), to_array() initial test failed", drone)
    print("DroneState: from_array(), to_array() initial test passed")

    # Test from_array(), to_array(), from_int(), to_int() using all possible states
    state_space_size = 0
    last_state = None
    for x in range(config.GRID_SIZE):
        for y in range(config.GRID_SIZE):
            for vx in range(config.VELOCITY_MIN, config.VELOCITY_MAX + 1):
                for vy in range(config.VELOCITY_MIN, config.VELOCITY_MAX + 1):
                    for battery in range(config.BATTERY_MIN, config.BATTERY_MAX + 1):
                        for wx in range(config.WIND_MIN, config.WIND_MAX + 1):
                            for wy in range(config.WIND_MIN, config.WIND_MAX + 1):
                                state_space_size += 1
                                ds = DroneState(
                                    position=np.array([x, y]), 
                                    velocity=np.array([vx, vy]), 
                                    battery=battery, 
                                    wind=np.array([wx, wy]))
                                last_state = ds
                                ds_array = ds.to_array()
                                ds_int = ds.to_int()
                                ds1 = DroneState.from_array(ds_array).to_array()
                                ds2 = DroneState.from_int(ds_int).to_array()
                                if not np.all(ds1 == ds2) or not np.all(ds1 == ds.to_array()):
                                    raise ValueError("DroneState: from_array(), to_array(), from_int(), to_int() all test failed", ds, ds1, ds2)
    print("DroneState: from_array(), to_array(), from_int(), to_int() all test passed")

    # Test is_terminal() from position
    for x in range(config.GRID_SIZE):
        for y in range(config.GRID_SIZE):
            drone.position = np.array([x, y])
            is_goal = np.array_equal([x, y], config.GOAL_POSITION)
            is_obstacle = any(np.array_equal([x, y], obs) for obs in config.OBSTACLES)
            if is_goal or is_obstacle:
                if not drone.is_terminal():
                    raise ValueError(f"DroneState: is_terminal() test failed at position {x}, {y}", drone)
                continue
            else:
                if drone.is_terminal():
                    raise ValueError(f"DroneState: is_terminal() test failed at position {x}, {y}", drone)
    print("DroneState: is_terminal() by position test passed")

    # Test get_state_space_size()
    if state_space_size != DroneState.get_state_space_size():
        raise ValueError("DroneState: get_state_space_size() test failed", state_space_size, DroneState.get_state_space_size())
    if state_space_size - 1 != last_state.to_int():
        raise ValueError("DroneState: get_state_space_size() test failed", last_state.to_int(), state_space_size - 1)
    print("DroneState: get_state_space_size() test passed: total states =", state_space_size)

    # Test get_state_space()
    state_space = DroneState.get_state_space()
    if len(state_space) != config.GRID_SIZE * config.GRID_SIZE * (config.VELOCITY_MAX - config.VELOCITY_MIN + 1) * (config.VELOCITY_MAX - config.VELOCITY_MIN + 1) * (config.BATTERY_MAX - config.BATTERY_MIN + 1) * (config.WIND_MAX - config.WIND_MIN + 1) * (config.WIND_MAX - config.WIND_MIN + 1):
        raise ValueError("DroneState: get_state_space() test failed", state_space)
    print("DroneState: get_state_space() test passed")

    # Test is_terminal() from battery
    drone.position = config.STARTING_POSITION
    for battery in range(config.BATTERY_MIN, config.BATTERY_MAX + 1):
        drone.battery = battery
        if battery <= config.BATTERY_MIN:
            if not drone.is_terminal():
                raise ValueError(f"DroneState: is_terminal() test failed at battery level {battery}", drone)
            continue
        else:
            if drone.is_terminal():
                raise ValueError(f"DroneState: is_terminal() test failed at battery level{battery}", drone)
    print("DroneState: is_terminal() by battery test passed")


if __name__ == "__main__":
    current_module = sys.modules[__name__]
    for name in dir(current_module):
        obj = getattr(current_module, name)
        if callable(obj) and name.endswith('_test') and name != "__main__":
            print(f"Running {name}()...")
            obj()