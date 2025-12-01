import random
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
                             current_robot_pos: Coordinate, num_particles: int = NUM_BELIEF_PARTICLES,
                             reinvigoration_ratio: float = 0.1) -> None:
        """
        CRITICAL FUNCTION (OPTIMIZED):
        1. Update Global Grid (Log Odds) based on Lidar.
        2. FAST incremental resampling: only resample cells that changed.
        3. Force all particles to be at 'current_robot_pos' (Perfect Localization).
        
        Args:
            reinvigoration_ratio: Fraction of particles to sample from scratch (for diversity)
        """
        
        # A. UPDATE GLOBAL MAP
        # Signal start of new observation for change tracking
        self.occupancy_grid.begin_observation()
        
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

        # B. RESAMPLE PARTICLES (OPTIMIZED - Incremental + Reinvigoration)
        new_particle_list = []
        
        # Get parent particles for incremental updates
        current_particles = list(self.belief.particles)
        num_reinvigorate = max(1, int(num_particles * reinvigoration_ratio))
        num_incremental = num_particles - num_reinvigorate
        
        # B1. INCREMENTAL PARTICLES (FAST) - inherit from parents, only resample changed cells
        for _ in range(num_incremental):
            # Pick a random parent particle
            parent = random.choice(current_particles)
            
            # Incremental sampling: only resample cells that changed in this observation
            sampled_obstacles = self.occupancy_grid.sample_incremental(parent._obstacle_set)
            
            new_particle_list.append(GridState(current_robot_pos, sampled_obstacles, self.goal))
        
        # B2. REINVIGORATION PARTICLES (for diversity) - sample from scratch
        for _ in range(num_reinvigorate):
            # Full sampling from the global grid (sparse - only observed cells)
            sampled_obstacles = self.occupancy_grid.sample_map_hypothesis()
            
            new_particle_list.append(GridState(current_robot_pos, sampled_obstacles, self.goal))
            
        # C. OVERRIDE BELIEF
        self.set_belief(pomdp_py.Particles(new_particle_list))