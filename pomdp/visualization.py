"""
Visualization module for POMDP drone navigation.
Uses matplotlib to display grid, obstacles, and robot movement.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import config
PLOT_FIGURE_SIZE = config.PLOT_FIGURE_SIZE

class GridVisualizer:
    """Real-time visualization of robot navigation on a grid."""
    
    def __init__(self, grid_width: int, grid_height: int, obstacles: list, goal, 
                 start_pos=None, agent=None, cell_size: float = 1.0):
        """
        Initialize the visualizer.
        
        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid  
            obstacles: List of Coordinate objects representing obstacles (true obstacles)
            goal: Coordinate object for goal position
            start_pos: Starting position (Coordinate object)
            agent: RobotAgent instance to access occupancy grid for belief visualization
            cell_size: Size of each cell for display
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles
        self.goal = goal
        self.start_pos = start_pos
        self.agent = agent
        self.cell_size = cell_size
        
        # Track path history and rewards
        self.path_history = []
        self.reward_history = []
        self.first_position_received = False
        
        # Initialize list to store dynamic artists for cleanup
        self._dynamic_artists = []
        
        # Store obstacle patches for updating opacity
        self.obstacle_patches = {}  # Dict[Coordinate, Rectangle]
        
        # Setup the plot
        self._setup_plot()
        
    def _setup_plot(self):
        """Initialize matplotlib figure and axes."""
        plt.ion()  # Enable interactive mode
        
        # Create figure matching MDP style
        self.fig, self.ax = plt.subplots(figsize=(self.grid_width, self.grid_height))
        
        # Set plot size
        self.fig.set_size_inches(PLOT_FIGURE_SIZE / self.fig.dpi, PLOT_FIGURE_SIZE / self.fig.dpi)
        
        # Plot obstacles as red squares with initial low opacity (10%)
        # Opacity will be updated based on agent's belief
        if len(self.obstacles) > 0:
            for i, obs in enumerate(self.obstacles):
                # Create 1x1 square centered at obstacle position
                rect = patches.Rectangle(
                    (obs.x - 0.5, obs.y - 0.5),  # Bottom-left corner
                    1.0, 1.0,  # Width and height (exactly 1x1)
                    linewidth=1,
                    edgecolor='darkred',
                    facecolor='red',
                    alpha=0.1,  # Start with 10% opacity (undiscovered)
                    zorder=3,
                    label='Obstacles' if i == 0 else ''  # Only add label once
                )
                self.ax.add_patch(rect)
                # Store reference to update opacity later
                self.obstacle_patches[obs] = rect
        
        # Plot goal as green star (matching MDP style)
        self.ax.scatter(self.goal.x, self.goal.y, 
                       c='green', s=300, marker='*', label='Goal', zorder=4, alpha=0.8,
                       edgecolors='blue', linewidths=2)
        
        # Initialize starting position marker (will be updated on first position)
        self.start_scatter = self.ax.scatter([], [], c='blue', s=150, marker='o', 
                                            label='Start', zorder=5, alpha=0.8,
                                            edgecolors='black', linewidths=1.5)
        
        # If start_pos provided, show it immediately
        if self.start_pos is not None:
            self.start_scatter.set_offsets([[self.start_pos.x, self.start_pos.y]])
        
        # Initialize path line (matching MDP style - blue line with cyan markers)
        self.path_line, = self.ax.plot([], [], color='blue', linewidth=2, alpha=0.6, 
                                        label='Drone Path', zorder=2)
        self.path_scatter = self.ax.scatter([], [], c='cyan', s=50, marker='o', alpha=0.7, 
                                            zorder=2, edgecolors='blue', linewidths=0.5)
        
        # Initialize current position marker (orange, matching MDP style)
        self.robot_scatter = self.ax.scatter([], [], c='orange', s=200, marker='o', 
                                            edgecolors='black', linewidths=2, zorder=6)
        
        # Set plot limits and labels (matching MDP style)
        self.ax.set_xlim(-0.5, self.grid_width - 0.5)
        self.ax.set_ylim(-0.5, self.grid_height - 0.5)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('POMDP Drone Path Visualization')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper left')
        self.ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)
    
    def update(self, robot_pos, step: int, action_name: str = "", reward: float = 0.0, total_reward: float = 0.0, elapsed_time: float = 0.0):
        """
        Update the visualization with new robot position.
        
        Args:
            robot_pos: Coordinate object with robot's current position
            step: Current simulation step
            action_name: Name of the action taken
            reward: Reward received at this step
            total_reward: Total accumulated reward
            elapsed_time: Elapsed time in seconds
        """
        # Remove all dynamic artists from previous update
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists = []
        
        # Track start position from first update if not already set
        if not self.first_position_received:
            if self.start_pos is None:
                # Set start position from first robot position
                self.start_scatter.set_offsets([[robot_pos.x, robot_pos.y]])
            self.first_position_received = True
        
        # Add to path history
        self.path_history.append((robot_pos.x, robot_pos.y))
        
        # Add reward to history (skip first step which has no reward)
        if step > 0:
            self.reward_history.append(reward)
        
        # Update path line and scatter
        if len(self.path_history) > 0:
            xs, ys = zip(*self.path_history)
            self.path_line.set_data(xs, ys)
            self.path_scatter.set_offsets(np.c_[xs, ys])
        
        # Add reward labels on each node (starting from position 1, as position 0 has no reward)
        if len(self.path_history) > 1:
            path_positions = np.array(self.path_history)
            for i, reward_val in enumerate(self.reward_history):
                pos = path_positions[i + 1]  # i+1 because reward_history[i] corresponds to transition to state[i+1]
                text = self.ax.text(pos[0], pos[1] + 0.5, f'{reward_val:.1f}', 
                                   fontsize=6, ha='center', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.4),
                                   zorder=7)
                self._dynamic_artists.append(text)
        
        # Update current position marker (orange circle)
        self.robot_scatter.set_offsets([[robot_pos.x, robot_pos.y]])
        
        # Update obstacle opacity based on agent's belief
        if self.agent is not None and hasattr(self.agent, 'occupancy_grid'):
            for obs_coord, rect_patch in self.obstacle_patches.items():
                # Get the agent's belief probability for this obstacle cell
                belief_prob = self.agent.occupancy_grid.get_probability(obs_coord)
                
                # Scale opacity: 10% baseline + up to 90% based on belief
                # belief_prob ranges from 0.0 (unknown/free) to 1.0 (certain obstacle)
                opacity = 0.1 + (belief_prob * 0.9)
                rect_patch.set_alpha(opacity)
        
        # Update title with step info (matching MDP format)
        dist = abs(robot_pos.x - self.goal.x) + abs(robot_pos.y - self.goal.y)
        reward_text = f'Reward: {reward:.2f}, ' if step > 0 else ''
        self.ax.set_title(f'POMDP Drone Path Visualization, Time: {elapsed_time:.2f}s\n'
                         f'Step: {step}, Action: {action_name}, {reward_text}Total Reward: {total_reward:.2f}\n'
                         f'Position: ({robot_pos.x}, {robot_pos.y}), Distance to Goal: {dist:.2f}')
        
        # Redraw
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause for animation effect
    
    def show_goal_reached(self):
        """Display goal reached celebration."""
        # Update title to show goal reached
        current_title = self.ax.get_title()
        self.ax.set_title(f'{current_title}')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the visualization window."""
        plt.ioff()
        plt.close(self.fig)
    
    def show(self):
        """Keep the plot window open (call at end of simulation)."""
        plt.ioff()
        plt.show()

