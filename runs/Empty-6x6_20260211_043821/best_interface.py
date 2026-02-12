"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Navigate to the goal tile in a 6x6 empty room.
Success rate: 46%
Obs dim:      15
Generation:   1
Iteration:    50
"""

import jax
import jax.numpy as jnp

def get_observation(state):
    """Simple but effective observation: agent position, direction, and goal direction."""
    H, W = state.grid.shape[:2]
    
    # Agent position normalized to [0, 1]
    agent_pos = state.agent.position.astype(jnp.float32)
    norm_pos = agent_pos / jnp.array([H - 1, W - 1], dtype=jnp.float32)
    
    # Agent direction as one-hot
    dir_onehot = jax.nn.one_hot(state.agent.direction, 4, dtype=jnp.float32)
    
    # Find goal position
    goal_mask = (state.grid[:, :, 0] == 6)
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    goal_count = jnp.maximum(jnp.sum(goal_mask), 1.0)
    goal_y = jnp.sum(yy * goal_mask) / goal_count
    goal_x = jnp.sum(xx * goal_mask) / goal_count
    
    # Goal position normalized
    norm_goal = jnp.array([goal_y, goal_x]) / jnp.array([H - 1, W - 1], dtype=jnp.float32)
    
    # Direction vector to goal (unnormalized)
    goal_direction = norm_goal - norm_pos
    
    # Distance to goal (normalized by max possible distance)
    max_dist = jnp.sqrt(2.0)  # diagonal of unit square
    goal_distance = jnp.sqrt(jnp.sum(goal_direction ** 2)) / max_dist
    
    # Simple local view: tiles in 4 cardinal directions from agent
    DIRECTIONS = jnp.array([(-1, 0), (0, 1), (1, 0), (0, -1)])  # Up, Right, Down, Left
    local_tiles = jnp.zeros(4, dtype=jnp.float32)
    
    for i in range(4):
        dy, dx = DIRECTIONS[i]
        neighbor_y = jnp.clip(agent_pos[0] + dy, 0, H - 1).astype(jnp.int32)
        neighbor_x = jnp.clip(agent_pos[1] + dx, 0, W - 1).astype(jnp.int32)
        
        # Check if position is in bounds
        in_bounds = ((agent_pos[0] + dy >= 0) & (agent_pos[0] + dy < H) & 
                    (agent_pos[1] + dx >= 0) & (agent_pos[1] + dx < W))
        
        tile_id = jnp.where(in_bounds, 
                           state.grid[neighbor_y, neighbor_x, 0].astype(jnp.float32),
                           2.0)  # WALL for out of bounds
        local_tiles = local_tiles.at[i].set(tile_id / 12.0)  # Normalize tile IDs
    
    # Concatenate all features
    observation = jnp.concatenate([
        norm_pos,           # 2 elements: normalized agent position
        dir_onehot,         # 4 elements: agent direction
        norm_goal,          # 2 elements: normalized goal position  
        goal_direction,     # 2 elements: direction vector to goal
        jnp.array([goal_distance]),  # 1 element: distance to goal
        local_tiles         # 4 elements: neighboring tiles
    ])
    
    return observation

def compute_reward(state, action, next_state):
    """Simple distance-based reward for navigation."""
    # Find goal position
    goal_mask = (state.grid[:, :, 0] == 6)
    H, W = state.grid.shape[:2]
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    goal_count = jnp.maximum(jnp.sum(goal_mask), 1.0)
    goal_y = jnp.sum(yy * goal_mask) / goal_count
    goal_x = jnp.sum(xx * goal_mask) / goal_count
    
    # Manhattan distance progress
    prev_pos = state.agent.position.astype(jnp.float32)
    curr_pos = next_state.agent.position.astype(jnp.float32)
    
    prev_dist = jnp.abs(prev_pos[0] - goal_y) + jnp.abs(prev_pos[1] - goal_x)
    curr_dist = jnp.abs(curr_pos[0] - goal_y) + jnp.abs(curr_pos[1] - goal_x)
    
    # Reward for getting closer to goal
    reward = prev_dist - curr_dist
    
    return reward.astype(jnp.float32)
