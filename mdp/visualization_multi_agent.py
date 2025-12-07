# This file was generated with Cursor/Claude 4.5 Sonnet

import numpy as np
import matplotlib.pyplot as plt

import config
from mdp.simulator_multi_agent import MultiAgentSimulator
from mdp.drone_multi_agent import MultiAgentDroneMDP

# Color scheme for different agents
AGENT_COLORS = ['blue', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
AGENT_LIGHT_COLORS = ['cyan', 'salmon', 'plum', 'peachpuff', 'tan', 'lightpink', 'lightgray', 'khaki']

def setup_live_plot_multi_agent(seed: int, mdp: MultiAgentDroneMDP = None, base_policy=None):
    """Set up the initial plot for live visualization of multi-agent system."""
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
    
    # Plot A* paths if available from base_policy
    if base_policy is not None and hasattr(base_policy, 'agent_mdps'):
        for agent_id in range(min(config.NUM_AGENTS, len(base_policy.agent_mdps))):
            agent_mdp = base_policy.agent_mdps[agent_id]
            if len(agent_mdp.shortest_path) > 0:  # Plot any path that exists
                path_positions = np.array(agent_mdp.shortest_path)
                agent_color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
                ax.plot(path_positions[:, 0], path_positions[:, 1], 
                        '--', linewidth=2, alpha=0.3, 
                        color=agent_color,
                        label=f'A* Path {agent_id}' if agent_id < 3 else '', zorder=1)
                ax.scatter(path_positions[:, 0], path_positions[:, 1], 
                           c=agent_color, s=20, marker='s', alpha=0.3, zorder=1)
    
    # Plot goals as green stars for each agent
    for agent_id in range(config.NUM_AGENTS):
        goal_pos = config.MULTI_AGENT_GOAL_POSITIONS[agent_id]
        ax.scatter(goal_pos[0], goal_pos[1], 
                   c='green', s=300, marker='*', 
                   label=f'Goal {agent_id}' if agent_id < 3 else '', 
                   zorder=4, alpha=0.8,
                   edgecolors=AGENT_COLORS[agent_id % len(AGENT_COLORS)],
                   linewidths=2)
    
    # Plot starting positions as circles for each agent
    for agent_id in range(config.NUM_AGENTS):
        start_pos = config.MULTI_AGENT_STARTING_POSITIONS[agent_id]
        ax.scatter(start_pos[0], start_pos[1], 
                   c=AGENT_COLORS[agent_id % len(AGENT_COLORS)], 
                   s=150, marker='o', 
                   label=f'Start {agent_id}' if agent_id < 3 else '', 
                   zorder=5, alpha=0.8,
                   edgecolors='black', linewidths=1.5)
    
    # Set plot limits and labels
    ax.set_xlim(-0.5, config.GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, config.GRID_SIZE - 0.5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Live Multi-Agent Drone Paths (Seed: {seed})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    
    # Initialize list to store dynamic artists for cleanup
    ax._dynamic_artists = []
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)
    
    return fig, ax

def update_live_plot_multi_agent(fig, ax, simulator: MultiAgentSimulator, rewards: np.ndarray, 
                                  seed: int, mdp: MultiAgentDroneMDP = None, elapsed_time: float = 0,
                                  base_policy=None):
    """Update the live plot with current multi-agent drone paths."""
    # Remove all dynamic artists from previous update
    if hasattr(ax, '_dynamic_artists'):
        for artist in ax._dynamic_artists:
            artist.remove()
    ax._dynamic_artists = []
    
    # Get paths for all agents
    for agent_id in range(config.NUM_AGENTS):
        agent_positions = np.array([
            state.get_agent_state(agent_id).position 
            for state in simulator.joint_states_history
        ])
        
        if len(agent_positions) > 1:
            # Plot agent path as a line with markers
            agent_color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
            agent_light_color = AGENT_LIGHT_COLORS[agent_id % len(AGENT_LIGHT_COLORS)]
            
            line = ax.plot(agent_positions[:, 0], agent_positions[:, 1], 
                          color=agent_color, linewidth=2, alpha=0.6, zorder=2)[0]
            ax._dynamic_artists.append(line)
            
            scatter = ax.scatter(agent_positions[:, 0], agent_positions[:, 1], 
                                c=agent_light_color, s=50, marker='o', alpha=0.7, zorder=2,
                                edgecolors=agent_color, linewidths=0.5)
            ax._dynamic_artists.append(scatter)
            
            # Add wind direction arrows at each position
            for state in simulator.joint_states_history:
                agent_state = state.get_agent_state(agent_id)
                wind = agent_state.wind
                pos = agent_state.position
                # Draw arrow from position pointing in wind direction (one unit long)
                quiver = ax.quiver(pos[0], pos[1], wind[0], wind[1], 
                                  color='purple', alpha=0.5, scale=1, scale_units='xy',
                                  angles='xy', width=0.006, headwidth=3, headlength=4, zorder=1)
                ax._dynamic_artists.append(quiver)
            
            # Add reward labels on each node (starting from position 1, as position 0 has no reward)
            for i, reward_vals in enumerate(simulator.reward_history):
                pos = agent_positions[i + 1]  # i+1 because reward_history[i] corresponds to state[i+1]
                reward_val = reward_vals[agent_id]
                text = ax.text(pos[0], pos[1] + 0.5, f'{reward_val:.1f}', 
                              fontsize=6, ha='center', va='bottom',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.4),
                              zorder=7)
                ax._dynamic_artists.append(text)
        
        # Highlight current position with larger marker
        if len(agent_positions) > 0:
            current_pos = agent_positions[-1]
            current_scatter = ax.scatter(current_pos[0], current_pos[1], 
                                        c='orange', s=200, marker='o', 
                                        edgecolors='black', 
                                        linewidths=2, zorder=6)
            ax._dynamic_artists.append(current_scatter)
    
    # Update title with current stats
    total_reward = np.sum(simulator.get_total_reward())
    cumulative_rewards = simulator.get_total_reward()
    
    # Get battery and wind info for each agent
    battery_text = ", ".join([f"A{i}:{simulator.joint_states.get_agent_state(i).battery}" 
                               for i in range(config.NUM_AGENTS)])
    # Wind is the same for all agents, so just get it from the first agent
    wind = simulator.joint_states.get_agent_state(0).wind
    wind_text = f"[{wind[0]:.1f},{wind[1]:.1f}]"
    
    ax.set_title(f'Live Multi-Agent Drone Paths (Seed: {seed}), Time: {elapsed_time:.2f}s\n'
                 f'Reward: {np.sum(rewards):.2f}, Cumulative Reward: {total_reward:.2f}\n'
                 f'Step: {simulator.tick}, Battery: [{battery_text}], Wind: {wind_text}')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

def visualize_path_multi_agent(simulator: MultiAgentSimulator, seed: int, 
                                mdp: MultiAgentDroneMDP = None, elapsed_time: float = 0,
                                base_policy=None):
    """Create a final detailed visualization of the complete multi-agent drone paths."""
    fig, ax = plt.subplots(figsize=(config.GRID_SIZE, config.GRID_SIZE))
    
    # Set plot size
    fig.set_size_inches(config.PLOT_FIGURE_SIZE / fig.dpi, config.PLOT_FIGURE_SIZE / fig.dpi)
    
    # Plot obstacles as red circles with size corresponding to OBSTACLE_THRESHOLD
    if len(config.OBSTACLES) > 0:
        points_per_grid_unit = config.PLOT_FIGURE_SIZE / config.GRID_SIZE
        obstacle_marker_size = ((config.OBSTACLE_THRESHOLD * points_per_grid_unit) ** 2) / 2
        ax.scatter(
            config.OBSTACLES[:, 0], config.OBSTACLES[:, 1],
            c='red', s=obstacle_marker_size, marker='o',
            label='Obstacles', zorder=3, alpha=0.6
        )
    
    # Plot A* paths if available from base_policy
    if base_policy is not None and hasattr(base_policy, 'agent_mdps'):
        for agent_id in range(min(config.NUM_AGENTS, len(base_policy.agent_mdps))):
            agent_mdp = base_policy.agent_mdps[agent_id]
            if len(agent_mdp.shortest_path) > 0:  # Plot any path that exists
                path_positions = np.array(agent_mdp.shortest_path)
                agent_color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
                ax.plot(path_positions[:, 0], path_positions[:, 1], 
                        '--', linewidth=2, alpha=0.3, 
                        color=agent_color,
                        label=f'A* Path {agent_id}', zorder=1)
                ax.scatter(path_positions[:, 0], path_positions[:, 1], 
                           c=agent_color, s=20, marker='s', alpha=0.3, zorder=1)
    
    # Plot goals as green stars for each agent
    for agent_id in range(config.NUM_AGENTS):
        goal_pos = config.MULTI_AGENT_GOAL_POSITIONS[agent_id]
        ax.scatter(goal_pos[0], goal_pos[1], 
                   c='green', s=300, marker='*', 
                   label=f'Goal {agent_id}', 
                   zorder=4, alpha=0.8,
                   edgecolors=AGENT_COLORS[agent_id % len(AGENT_COLORS)],
                   linewidths=2)
    
    # Plot starting positions for each agent
    for agent_id in range(config.NUM_AGENTS):
        start_pos = config.MULTI_AGENT_STARTING_POSITIONS[agent_id]
        ax.scatter(start_pos[0], start_pos[1], 
                   c=AGENT_COLORS[agent_id % len(AGENT_COLORS)], 
                   s=150, marker='o', 
                   label=f'Start {agent_id}', 
                   zorder=5, alpha=0.8,
                   edgecolors='black', linewidths=1.5)
    
    # Plot paths for all agents
    wind_added_to_legend = False
    for agent_id in range(config.NUM_AGENTS):
        agent_positions = np.array([
            state.get_agent_state(agent_id).position 
            for state in simulator.joint_states_history
        ])
        
        agent_color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
        agent_light_color = AGENT_LIGHT_COLORS[agent_id % len(AGENT_LIGHT_COLORS)]
        
        # Plot agent path as a line with markers
        ax.plot(agent_positions[:, 0], agent_positions[:, 1], 
                color=agent_color, linewidth=2, alpha=0.6, 
                label=f'Agent {agent_id} Path', zorder=2)
        ax.scatter(agent_positions[:, 0], agent_positions[:, 1], 
                   c=agent_light_color, s=50, marker='o', alpha=0.7, zorder=2,
                   edgecolors=agent_color, linewidths=0.5)
        
        # Add wind direction arrows at each position
        for state in simulator.joint_states_history:
            agent_state = state.get_agent_state(agent_id)
            wind = agent_state.wind
            pos = agent_state.position
            # Draw arrow from position pointing in wind direction (one unit long)
            ax.quiver(pos[0], pos[1], wind[0], wind[1], 
                     color='purple', alpha=0.5, scale=1, scale_units='xy',
                     angles='xy', width=0.006, headwidth=3, headlength=4, zorder=1,
                     label='Wind' if not wind_added_to_legend else '')
            wind_added_to_legend = True
        
        # Add reward labels on each node (starting from position 1, as position 0 has no reward)
        for i, reward_vals in enumerate(simulator.reward_history):
            pos = agent_positions[i + 1]  # i+1 because reward_history[i] corresponds to state[i+1]
            reward_val = reward_vals[agent_id]
            ax.text(pos[0], pos[1] + 0.5, f'{reward_val:.1f}', 
                   fontsize=6, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.4),
                   zorder=7)
    
    # Set plot limits and labels
    ax.set_xlim(-0.5, config.GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, config.GRID_SIZE - 0.5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Create title with summary statistics
    total_rewards = simulator.get_total_reward()
    reward_summary = ", ".join([f"A{i}:{total_rewards[i]:.1f}" for i in range(config.NUM_AGENTS)])
    total_system_reward = np.sum(total_rewards)
    
    ax.set_title(f'Multi-Agent Drone Path Visualization (Seed: {seed})\n'
                 f'Agent Rewards: [{reward_summary}], System Total: {total_system_reward:.2f}\n'
                 f'Total Steps: {len(simulator.joint_states_history)}, Time: {elapsed_time:.2f}s')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

