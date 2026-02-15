"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Navigate to the goal tile in a 6x6 empty room.
Success rate: 100%
Obs dim:      11
Generation:   0
Iteration:    3
Mode:         full
"""

import jax
import jax.numpy as jnp

def get_observation(state):
    """Extract observation for navigation task."""
    H, W = state.grid.shape[:2]
    
    # Agent position (normalized to [0, 1])
    agent_pos = state.agent.position.astype(jnp.float32)
    norm_agent_y = agent_pos[0] / (H - 1)
    norm_agent_x = agent_pos[1] / (W - 1)
    
    # Agent direction (one-hot encoded)
    direction_onehot = jax.nn.one_hot(state.agent.direction, 4, dtype=jnp.float32)
    
    # Find goal position
    goal_mask = (state.grid[:, :, 0] == 6)  # GOAL tile type
    
    # Get goal coordinates using meshgrid
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    # Calculate goal position (weighted average if multiple goals)
    goal_count = jnp.sum(goal_mask)
    goal_y = jnp.sum(yy * goal_mask) / jnp.maximum(goal_count, 1.0)
    goal_x = jnp.sum(xx * goal_mask) / jnp.maximum(goal_count, 1.0)
    
    # Normalize goal position
    norm_goal_y = goal_y / (H - 1)
    norm_goal_x = goal_x / (W - 1)
    
    # Distance to goal (Manhattan distance, normalized)
    manhattan_dist = jnp.abs(agent_pos[0] - goal_y) + jnp.abs(agent_pos[1] - goal_x)
    max_dist = (H - 1) + (W - 1)  # Maximum possible Manhattan distance
    norm_manhattan_dist = manhattan_dist / max_dist
    
    # Euclidean distance (normalized)
    euclidean_dist = jnp.sqrt((agent_pos[0] - goal_y)**2 + (agent_pos[1] - goal_x)**2)
    max_euclidean = jnp.sqrt((H - 1)**2 + (W - 1)**2)
    norm_euclidean_dist = euclidean_dist / max_euclidean
    
    # Step count (normalized)
    norm_step = state.step_num.astype(jnp.float32) / 100.0
    
    # Concatenate all features
    obs = jnp.concatenate([
        jnp.array([norm_agent_y, norm_agent_x]),  # Agent position (2)
        direction_onehot,                         # Agent direction (4)
        jnp.array([norm_goal_y, norm_goal_x]),    # Goal position (2)
        jnp.array([norm_manhattan_dist]),         # Manhattan distance (1)
        jnp.array([norm_euclidean_dist]),         # Euclidean distance (1)
        jnp.array([norm_step])                    # Step count (1)
    ])
    
    return obs

def compute_reward(state, action, next_state):
    """Compute reward for navigation task."""
    # Check if goal is reached in next state
    agent_pos = next_state.agent.position
    goal_tile = next_state.grid[agent_pos[0], agent_pos[1], 0]
    goal_reached = (goal_tile == 6).astype(jnp.float32)
    
    # Calculate distances for progress shaping
    H, W = state.grid.shape[:2]
    
    # Find goal position
    goal_mask = (state.grid[:, :, 0] == 6)
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    goal_count = jnp.sum(goal_mask)
    goal_y = jnp.sum(yy * goal_mask) / jnp.maximum(goal_count, 1.0)
    goal_x = jnp.sum(xx * goal_mask) / jnp.maximum(goal_count, 1.0)
    
    # Previous distance (before action)
    prev_pos = state.agent.position.astype(jnp.float32)
    prev_dist = jnp.sqrt((prev_pos[0] - goal_y)**2 + (prev_pos[1] - goal_x)**2)
    
    # Current distance (after action)
    curr_pos = next_state.agent.position.astype(jnp.float32)
    curr_dist = jnp.sqrt((curr_pos[0] - goal_y)**2 + (curr_pos[1] - goal_x)**2)
    
    # Progress reward (positive for getting closer, negative for moving away)
    progress_reward = (prev_dist - curr_dist) * 1.0
    
    # Small step penalty to encourage efficiency
    step_penalty = -0.01
    
    # Large completion bonus
    completion_bonus = goal_reached * 10.0
    
    # Total reward
    reward = progress_reward + step_penalty + completion_bonus
    
    return reward
