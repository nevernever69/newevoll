"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Navigate to the goal tile in a 6x6 empty room.
Success rate: 42%
Obs dim:      6
Generation:   1
Iteration:    20
"""

import jax
import jax.numpy as jnp

def get_observation(state):
    """Extremely minimal observation - just 4 core features."""
    # Get grid dimensions
    H, W = state.grid.shape[:2]
    
    # Agent position (normalized)
    agent_pos = state.agent.position.astype(jnp.float32)
    norm_agent_y = agent_pos[0] / (H - 1)
    norm_agent_x = agent_pos[1] / (W - 1)
    
    # Find goal position
    goal_mask = (state.grid[:, :, 0] == 6)
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    goal_count = jnp.sum(goal_mask)
    goal_y = jnp.sum(yy * goal_mask) / jnp.maximum(goal_count, 1.0)
    goal_x = jnp.sum(xx * goal_mask) / jnp.maximum(goal_count, 1.0)
    
    # Direction to goal (normalized displacement)
    rel_y = (goal_y - agent_pos[0]) / (H - 1)
    rel_x = (goal_x - agent_pos[1]) / (W - 1)
    
    # Agent direction encoded as sine/cosine for smooth rotation
    angle = state.agent.direction.astype(jnp.float32) * jnp.pi / 2.0  # 0, π/2, π, 3π/2
    dir_sin = jnp.sin(angle)
    dir_cos = jnp.cos(angle)
    
    # Ultra-minimal 6 features - position + goal direction + facing direction
    observation = jnp.array([
        norm_agent_y,  # Agent Y position
        norm_agent_x,  # Agent X position
        rel_y,         # Direction to goal Y
        rel_x,         # Direction to goal X  
        dir_sin,       # Facing direction (sine component)
        dir_cos        # Facing direction (cosine component)
    ])
    
    return observation.astype(jnp.float32)

def compute_reward(state, action, next_state):
    """Pure Manhattan distance shaping - single clean signal."""
    # Find goal position
    goal_mask = (state.grid[:, :, 0] == 6)
    
    H, W = state.grid.shape[:2]
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    goal_count = jnp.sum(goal_mask)
    goal_y = jnp.sum(yy * goal_mask) / jnp.maximum(goal_count, 1.0)
    goal_x = jnp.sum(xx * goal_mask) / jnp.maximum(goal_count, 1.0)
    
    # Agent positions
    prev_pos = state.agent.position.astype(jnp.float32)
    curr_pos = next_state.agent.position.astype(jnp.float32)
    
    # Manhattan distances (grid-aligned, discrete-friendly)
    prev_dist = jnp.abs(prev_pos[0] - goal_y) + jnp.abs(prev_pos[1] - goal_x)
    curr_dist = jnp.abs(curr_pos[0] - goal_y) + jnp.abs(curr_pos[1] - goal_x)
    
    # Single reward term - pure distance improvement
    reward = prev_dist - curr_dist
    
    return reward.astype(jnp.float32)
