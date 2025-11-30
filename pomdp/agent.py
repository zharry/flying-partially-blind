import pomdp_py
from state_observation import *
from config import GRID_WIDTH, GRID_HEIGHT, DIRECTIONS, NUM_BELIEF_PARTICLES

class RobotAgent(pomdp_py.Agent):
    def __init__(self, init_pos: Coordinate, goal: Coordinate, directions):
        self.directions = directions
        self.goal = goal
        self.occupancy_grid = GlobalOccupancyGrid(GRID_WIDTH, GRID_HEIGHT)
        
        # Initialize belief: Robot knows where it is.
        # But map is empty (or 50/50). User said "initially no obstacles".
        # Create separate GridState objects to avoid shared reference issues
        init_belief = pomdp_py.Particles([
            GridState(init_pos, obstacle_pos=[], goal=goal) for _ in range(200)
        ])
        
        super().__init__(init_belief, 
                         GridPolicyModel(ACTIONS),  # Custom policy model with all actions
                         GridTransitionModel(),
                         GridObservationModel(directions),
                         GridRewardModel())

    def manual_belief_update(self, real_action: GridAction, real_observation: LidarObservation, 
                             current_robot_pos: Coordinate, num_particles: int = NUM_BELIEF_PARTICLES) -> None:
        """
        CRITICAL FUNCTION:
        1. Update Global Grid (Log Odds) based on Lidar.
        2. Resample completely new particles from the Global Grid.
        3. Force all particles to be at 'current_robot_pos' (Perfect Localization).
        """
        
        # A. UPDATE GLOBAL MAP
        # We use the known ground-truth position (current_robot_pos) as the origin
        # for the Lidar scan we just received.
        rx, ry = current_robot_pos.x, current_robot_pos.y
            
        # Map Lidar Beams to World Coordinates
        for beam_idx, beam in enumerate(real_observation.beams):
            dx, dy = self.directions[beam_idx]
            cx, cy = rx, ry
            
            for depth, is_blocked in enumerate(beam):
                cx += dx
                cy += dy
                
                if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                    # UPDATE THE GLOBAL LOG-ODDS GRID
                    # 1 = Blocked, 0 = Free
                    coord = Coordinate(cx, cy)
                    self.occupancy_grid.update(coord, is_blocked == 1)

        # B. RESAMPLE PARTICLES (Reinvigoration)
        new_particle_list = []
        
        for _ in range(num_particles):
            # 1. Sample a map from the updated Global Grid
            sampled_obstacles = self.occupancy_grid.sample_map_hypothesis()
            
            # 2. Create state. 
            # We enforce perfect localization: The particle MUST be where the robot is.
            # We do NOT check for wall collisions here because if the real robot 
            # is standing there, the map hypothesis must be wrong if it thinks there's a wall.
            # (Or we just accept the inconsistency for the sake of planning)
            
            # Optimization: If the sampled map puts a wall EXACTLY where we are standing,
            # we should technically reject that map, but for simplicity we keep it 
            # so the planner learns "I am stuck" if necessary.
            
            new_particle_list.append(GridState(current_robot_pos, sampled_obstacles, self.goal))
            
        # C. OVERRIDE BELIEF
        self.set_belief(pomdp_py.Particles(new_particle_list))