# This file was generated with Cursor/Claude 4.5 Sonnet

import numpy as np
import matplotlib.pyplot as plt

import config
from mdp.simulator import Simulator
from mdp.drone import DroneMDP

def setup_live_plot(seed: int, mdp: DroneMDP = None):
    """Set up the initial plot for live visualization."""
    fig, ax = plt.subplots(figsize=(config.GRID_SIZE, config.GRID_SIZE))

    # Set plot size
    fig.set_size_inches(config.PLOT_FIGURE_SIZE / fig.dpi, config.PLOT_FIGURE_SIZE / fig.dpi)
    
    # Plot obstacles as red circles
    if len(config.OBSTACLES) > 0:
        points_per_grid_unit = config.PLOT_FIGURE_SIZE / config.GRID_SIZE
        obstacle_marker_size = ((config.OBSTACLE_THRESHOLD * points_per_grid_unit) ** 2) / 2
        ax.scatter(
            config.OBSTACLES[:, 0], config.OBSTACLES[:, 1],
            c='red', s=obstacle_marker_size, marker='o',
            label='Obstacles', zorder=3, alpha=0.6
        )
    
    # Plot goal as green star
    ax.scatter(config.GOAL_POSITION[0], config.GOAL_POSITION[1], 
               c='green', s=300, marker='*', label='Goal', zorder=4)
    
    # Plot starting position as blue circle
    ax.scatter(config.STARTING_POSITION[0][0], config.STARTING_POSITION[0][1], 
               c='blue', s=150, marker='o', label='Start', zorder=5)
    
    # Plot A* path if available
    if mdp is not None and len(mdp.shortest_path) > 1:
        path_positions = np.array(mdp.shortest_path)
        ax.plot(path_positions[:, 0], path_positions[:, 1], 
                'g--', linewidth=2, alpha=0.5, label='A* Path', zorder=1)
        ax.scatter(path_positions[:, 0], path_positions[:, 1], 
                   c='lightgreen', s=30, marker='s', alpha=0.5, zorder=1)
    
    # Set plot limits and labels
    ax.set_xlim(-0.5, config.GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, config.GRID_SIZE - 0.5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Live Drone Path (Seed: {seed})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)
    
    return fig, ax

def update_live_plot(fig, ax, simulator: Simulator, reward: float, seed: int, mdp: DroneMDP = None, elapsed_time: float = 0):
    """Update the live plot with current drone path."""
    # Clear previous path lines and text annotations (but keep obstacles, goal, start, and optionally A* path)
    # Remove all lines, path markers, and text from previous updates
    num_keep = 3  # obstacles, goal, start
    if mdp is not None and len(mdp.shortest_path) > 1:
        num_keep = 5  # also keep A* path line and markers
    for artist in ax.lines + ax.collections[num_keep:]:
        artist.remove()
    
    # Remove text annotations
    for text in ax.texts[:]:
        text.remove()
    
    # Get current path
    drone_positions = np.array([state.position for state in simulator.state_history])
    
    if len(drone_positions) > 1:
        # Plot drone path as a line with markers
        ax.plot(drone_positions[:, 0], drone_positions[:, 1], 
                'b-', linewidth=2, alpha=0.6, zorder=2)
        ax.scatter(drone_positions[:, 0], drone_positions[:, 1], 
                   c='cyan', s=50, marker='o', alpha=0.7, zorder=2)
        
        # Add wind direction arrows at each position
        for state in simulator.state_history:
            wind = state.wind
            pos = state.position
            # Draw arrow from position pointing in wind direction (one unit long)
            ax.quiver(pos[0], pos[1], wind[0], wind[1], 
                     color='purple', alpha=0.5, scale=1, scale_units='xy',
                     angles='xy', width=0.006, headwidth=3, headlength=4, zorder=1)
        
        # Add reward labels on each node (starting from position 1, as position 0 has no reward)
        for i, reward_val in enumerate(simulator.reward_history):
            pos = drone_positions[i + 1]  # i+1 because reward_history[i] corresponds to transition to state[i+1]
            ax.text(pos[0], pos[1] + 0.5, f'{reward_val:.1f}', 
                   fontsize=6, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.4),
                   zorder=7)
    
    # Highlight current position with larger marker
    if len(drone_positions) > 0:
        current_pos = drone_positions[-1]
        ax.scatter(current_pos[0], current_pos[1], 
                   c='orange', s=200, marker='o', edgecolors='black', 
                   linewidths=2, zorder=6, label='Current')
    
    # Update title with current stats
    ax.set_title(f'Live Drone Path (Seed: {seed}), Time: {elapsed_time:.2f}s\n'
                 f'Reward: {reward:.2f}, Cumulative Reward: {simulator.get_total_reward():.2f}\n'
                 f'Step: {simulator.tick}, Battery: {simulator.state.battery}, Wind: [{simulator.state.wind[0]:.1f}, {simulator.state.wind[1]:.1f}]')
    
    plt.tight_layout()  # Adjust layout to prevent text cutoff
    plt.draw()
    plt.pause(0.01)  # Small pause to allow plot to update

def visualize_path(simulator: Simulator, seed: int, mdp: DroneMDP = None, elapsed_time: float = 0):
    """Create a final detailed visualization of the complete drone path."""
    fig, ax = plt.subplots(figsize=(config.GRID_SIZE, config.GRID_SIZE))
    
    # Set plot size
    fig.set_size_inches(config.PLOT_FIGURE_SIZE / fig.dpi, config.PLOT_FIGURE_SIZE / fig.dpi)
    
    drone_positions = np.array([state.position for state in simulator.state_history])
    
    # Plot obstacles as red circles with size corresponding to OBSTACLE_THRESHOLD
    if len(config.OBSTACLES) > 0:
        points_per_grid_unit = config.PLOT_FIGURE_SIZE / config.GRID_SIZE
        obstacle_marker_size = ((config.OBSTACLE_THRESHOLD * points_per_grid_unit) ** 2) / 2
        ax.scatter(
            config.OBSTACLES[:, 0], config.OBSTACLES[:, 1],
            c='red', s=obstacle_marker_size, marker='o',
            label='Obstacles', zorder=3, alpha=0.6
        )
    
    # Plot goal as green star
    ax.scatter(config.GOAL_POSITION[0], config.GOAL_POSITION[1], 
               c='green', s=300, marker='*', label='Goal', zorder=4)
    
    # Plot starting position as blue circle
    ax.scatter(drone_positions[0, 0], drone_positions[0, 1], 
               c='blue', s=150, marker='o', label='Start', zorder=5)
    
    # Plot A* path if available
    if mdp is not None and len(mdp.shortest_path) > 1:
        path_positions = np.array(mdp.shortest_path)
        ax.plot(path_positions[:, 0], path_positions[:, 1], 
                'g--', linewidth=2, alpha=0.5, label='A* Path', zorder=1)
        ax.scatter(path_positions[:, 0], path_positions[:, 1], 
                   c='lightgreen', s=30, marker='s', alpha=0.5, zorder=1)
    
    # Plot drone path as a line with markers
    ax.plot(drone_positions[:, 0], drone_positions[:, 1], 
            'b-', linewidth=2, alpha=0.6, label='Drone Path', zorder=2)
    ax.scatter(drone_positions[:, 0], drone_positions[:, 1], 
               c='cyan', s=50, marker='o', alpha=0.7, zorder=2)
    
    # Add wind direction arrows at each position
    wind_added_to_legend = False
    for state in simulator.state_history:
        wind = state.wind
        pos = state.position
        # Draw arrow from position pointing in wind direction (one unit long)
        ax.quiver(pos[0], pos[1], wind[0], wind[1], 
                 color='purple', alpha=0.5, scale=1, scale_units='xy',
                 angles='xy', width=0.006, headwidth=3, headlength=4, zorder=1,
                 label='Wind' if not wind_added_to_legend else '')
        wind_added_to_legend = True
    
    # Add reward labels on each node (starting from position 1, as position 0 has no reward)
    for i, reward_val in enumerate(simulator.reward_history):
        pos = drone_positions[i + 1]  # i+1 because reward_history[i] corresponds to transition to state[i+1]
        ax.text(pos[0], pos[1] + 0.5, f'{reward_val:.1f}', 
               fontsize=6, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.4),
               zorder=7)
    
    # Set plot limits and labels
    ax.set_xlim(-0.5, config.GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, config.GRID_SIZE - 0.5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Drone Path Visualization (Seed: {seed})\nTotal Reward: {simulator.get_total_reward():.2f}, Total Steps: {len(simulator.state_history)}, Time: {elapsed_time:.2f}s')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

