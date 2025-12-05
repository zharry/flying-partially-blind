"""
Visualization module for POMDP drone navigation.
Uses matplotlib to display grid, obstacles, and robot movement.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

class GridVisualizer:
    """Real-time visualization of robot navigation on a grid."""
    
    def __init__(self, grid_width: int, grid_height: int, obstacles: list, goal, 
                 cell_size: float = 1.0):
        """
        Initialize the visualizer.
        
        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid  
            obstacles: List of Coordinate objects representing obstacles
            goal: Coordinate object for goal position
            cell_size: Size of each cell for display
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles
        self.goal = goal
        self.cell_size = cell_size
        
        # Track path history
        self.path_history = []
        
        # Setup the plot
        self._setup_plot()
        
    def _setup_plot(self):
        """Initialize matplotlib figure and axes."""
        plt.ion()  # Enable interactive mode
        
        # Scale figure size based on grid dimensions (cap at reasonable size)
        fig_size = min(12, max(8, self.grid_width // 4))
        
        # Create figure with dark theme
        self.fig, self.ax = plt.subplots(figsize=(fig_size, fig_size), facecolor='#1a1a2e')
        self.ax.set_facecolor('#16213e')
        
        # Set axis limits and aspect
        self.ax.set_xlim(-0.5, self.grid_width - 0.5)
        self.ax.set_ylim(-0.5, self.grid_height - 0.5)
        self.ax.set_aspect('equal')
        
        # Draw grid lines (reduce density for large grids)
        grid_step = 1 if self.grid_width <= 20 else (5 if self.grid_width <= 50 else 10)
        line_alpha = 0.7 if self.grid_width <= 20 else 0.4
        
        for i in range(0, self.grid_width + 1, grid_step):
            self.ax.axvline(x=i - 0.5, color='#0f3460', linewidth=0.5, alpha=line_alpha)
        for i in range(0, self.grid_height + 1, grid_step):
            self.ax.axhline(y=i - 0.5, color='#0f3460', linewidth=0.5, alpha=line_alpha)
        
        # Scale sizes based on grid size for visibility
        is_large_grid = self.grid_width > 20
        obstacle_size = 0.9 if is_large_grid else 0.8
        marker_radius = 0.4 if is_large_grid else 0.35
        robot_radius = 0.35 if is_large_grid else 0.3
        line_width = 1.5 if is_large_grid else 2
        font_size = 12 if is_large_grid else 16
        path_marker_size = 2 if is_large_grid else 4
        
        # Draw obstacles
        for obs in self.obstacles:
            obstacle_rect = patches.FancyBboxPatch(
                (obs.x - obstacle_size/2, obs.y - obstacle_size/2), obstacle_size, obstacle_size,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor='#e94560', edgecolor='#ff6b6b', linewidth=line_width,
                alpha=0.9
            )
            self.ax.add_patch(obstacle_rect)
        
        # Draw goal with glow effect
        goal_glow = plt.Circle(
            (self.goal.x, self.goal.y), marker_radius + 0.15, 
            color='#00ff88', alpha=0.3
        )
        self.ax.add_patch(goal_glow)
        
        goal_marker = plt.Circle(
            (self.goal.x, self.goal.y), marker_radius,
            facecolor='#00ff88', edgecolor='#00cc6a', linewidth=line_width
        )
        self.ax.add_patch(goal_marker)
        self.ax.text(self.goal.x, self.goal.y, '★', fontsize=font_size, 
                     ha='center', va='center', color='#1a1a2e', fontweight='bold')
        
        # Initialize robot marker (will be updated)
        self.robot_marker = plt.Circle(
            (0, 0), robot_radius, facecolor='#4cc9f0', edgecolor='#00b4d8', linewidth=line_width
        )
        self.ax.add_patch(self.robot_marker)
        
        # Initialize path line
        self.path_line, = self.ax.plot([], [], 'o-', color='#4cc9f0', 
                                        alpha=0.4, linewidth=line_width, markersize=path_marker_size)
        
        # Title and labels
        self.title = self.ax.set_title('POMDP Drone Navigation', 
                                        fontsize=16, fontweight='bold', 
                                        color='#eaeaea', pad=15)
        
        self.ax.set_xlabel('X', fontsize=12, color='#aaaaaa')
        self.ax.set_ylabel('Y', fontsize=12, color='#aaaaaa')
        
        # Customize ticks (reduce density for large grids)
        tick_step = 1 if self.grid_width <= 15 else (5 if self.grid_width <= 50 else 10)
        self.ax.set_xticks(range(0, self.grid_width, tick_step))
        self.ax.set_yticks(range(0, self.grid_height, tick_step))
        self.ax.tick_params(colors='#aaaaaa')
        for spine in self.ax.spines.values():
            spine.set_color('#0f3460')
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='#4cc9f0', edgecolor='#00b4d8', label='Robot'),
            patches.Patch(facecolor='#e94560', edgecolor='#ff6b6b', label='Obstacle'),
            patches.Patch(facecolor='#00ff88', edgecolor='#00cc6a', label='Goal'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', 
                       facecolor='#1a1a2e', edgecolor='#0f3460',
                       labelcolor='#eaeaea')
        
        # Status text
        self.status_text = self.ax.text(
            0.98, 0.02, '', transform=self.ax.transAxes,
            fontsize=10, color='#eaeaea', ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#0f3460', alpha=0.8)
        )
        
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.1)
    
    def update(self, robot_pos, step: int, action_name: str = ""):
        """
        Update the visualization with new robot position.
        
        Args:
            robot_pos: Coordinate object with robot's current position
            step: Current simulation step
            action_name: Name of the action taken
        """
        # Update robot marker position
        self.robot_marker.center = (robot_pos.x, robot_pos.y)
        
        # Add to path history
        self.path_history.append((robot_pos.x, robot_pos.y))
        
        # Update path line
        if len(self.path_history) > 1:
            xs, ys = zip(*self.path_history)
            self.path_line.set_data(xs, ys)
        
        # Update title with step info
        self.title.set_text(f'POMDP Drone Navigation | Step: {step} | Action: {action_name}')
        
        # Update status text
        status = f'Position: ({robot_pos.x}, {robot_pos.y})\nGoal: ({self.goal.x}, {self.goal.y})'
        dist = abs(robot_pos.x - self.goal.x) + abs(robot_pos.y - self.goal.y)
        status += f'\nManhattan Distance: {dist}'
        self.status_text.set_text(status)
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)  # Small pause for animation effect
    
    def show_goal_reached(self):
        """Display goal reached celebration."""
        self.title.set_text('🎉 GOAL REACHED! 🎉')
        self.title.set_color('#00ff88')
        
        # Add celebration effect
        celebration = plt.Circle(
            (self.goal.x, self.goal.y), 0.6,
            facecolor='none', edgecolor='#00ff88', linewidth=3, alpha=0.8
        )
        self.ax.add_patch(celebration)
        
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

