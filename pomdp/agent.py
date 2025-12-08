import sys
sys.path.append("..")

import random
import pomdp_py

from config import GRID_WIDTH, GRID_HEIGHT, DIRECTIONS
from pomdp.state_observation import *

class RobotAgent(pomdp_py.Agent):
    def __init__(self, init_pos: Coordinate, goal: Coordinate, directions):
        self.directions = directions
        self.goal = goal
        self.occupancy_grid = GlobalOccupancyGrid(GRID_WIDTH, GRID_HEIGHT)
        
        # Create shared A* pathfinder for reward model and rollout policy
        self.pathfinder = AStarPathfinder(GRID_WIDTH, GRID_HEIGHT, ACTIONS)
        
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
                         GridRewardModel(self.pathfinder))

    def manual_belief_update(self, real_action: GridAction, real_observation: LidarObservation, 
                             current_robot_pos: Coordinate) -> None:
        """
        Update belief particles based on new observation:
        1. Update Global Grid (Log Odds) based on Lidar.
        2. For each observed cell, update ALL particles in parallel:
           - Sample whether obstacle exists based on cell probability
           - Add/remove from particle's obstacle set
        3. Update all particle positions to current_robot_pos (Perfect Localization).
        """
        
        # A. UPDATE GLOBAL MAP
        self.occupancy_grid.begin_observation()
        
        rx, ry = current_robot_pos.x, current_robot_pos.y
            
        # Map Lidar Beams to World Coordinates
        for beam_idx, beam in enumerate(real_observation.beams):
            dx, dy = self.directions[beam_idx]
            cx, cy = rx, ry
            
            for depth, is_blocked in enumerate(beam):
                cx += dx
                cy += dy
                
                if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                    coord = Coordinate(cx, cy)
                    self.occupancy_grid.update(coord, is_blocked == 1)

        # B. UPDATE ALL PARTICLES - For each observed cell, update all particles
        current_particles = list(self.belief.particles)
        
        # Convert each particle's obstacles to a mutable set for efficient add/remove
        particle_obstacle_sets = [set(p._obstacle_set) for p in current_particles]
        
        # For each cell observed in this step, update all particles in parallel
        for coord in self.occupancy_grid.last_updated_cells:
            prob_occupied = self.occupancy_grid.get_probability(coord)
            
            # Update all particles for this cell
            for obs_set in particle_obstacle_sets:
                if random.random() < prob_occupied:
                    obs_set.add(coord)
                else:
                    obs_set.discard(coord)
        
        # C. CREATE UPDATED PARTICLES with new position and updated obstacles
        new_particle_list = [
            GridState(current_robot_pos, list(obs_set), self.goal)
            for obs_set in particle_obstacle_sets
        ]
        
        # D. OVERRIDE BELIEF
        self.set_belief(pomdp_py.Particles(new_particle_list))