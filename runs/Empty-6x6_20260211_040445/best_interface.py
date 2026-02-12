"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Navigate to the goal tile in a 6x6 empty room.
Success rate: 40%
Obs dim:      10
Generation:   0
Iteration:    4
"""

import jax
import jax.numpy as jnp

def get_observation(state):
    """Extract observation for navigation task.
    
    Features:
    - Agent position (normalized to [0,1])
    - Agent direction (one-hot encoded)  
    - Goal position (normalized to [0,1])
    - Distance to goal (normalized)
    - Step count (normalized)
    
    Total: 2 + 4 + 2 + 1 + 1 = 10 features
    """
    H, W = state.grid.shape[:2]
    
    # Agent position normalized to [0, 1]
    agent_pos_norm = state.agent.position.astype(jnp.float32) / jnp.array([H-1, W-1], dtype=jnp.float32)
    
    # Agent direction as one-hot (4 directions)
    agent_dir_onehot = jax.nn.one_hot(state.agent.direction, 4, dtype=jnp.float32)
    
    # Find goal position
    goal_mask = (state.grid[:, :, 0] == 6)  # GOAL tile ID is 6
    
    # Get goal coordinates using meshgrid
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    # Mean position of goal tiles (handles case of multiple goals)
    goal_count = jnp.maximum(jnp.sum(goal_mask), 1.0)  # Avoid division by zero
    goal_y = jnp.sum(yy * goal_mask) / goal_count
    goal_x = jnp.sum(xx * goal_mask) / goal_count
    
    # Goal position normalized to [0, 1]
    goal_pos_norm = jnp.array([goal_y, goal_x]) / jnp.array([H-1, W-1], dtype=jnp.float32)
    
    # Manhattan distance to goal (normalized by max possible distance)
    agent_pos_float = state.agent.position.astype(jnp.float32)
    manhattan_dist = jnp.abs(agent_pos_float[0] - goal_y) + jnp.abs(agent_pos_float[1] - goal_x)
    max_dist = (H - 1) + (W - 1)  # Max possible Manhattan distance
    dist_norm = manhattan_dist / max_dist
    
    # Step count normalized (assuming max steps around 100-200)
    step_norm = state.step_num.astype(jnp.float32) / 100.0
    
    # Concatenate all features
    obs = jnp.concatenate([
        agent_pos_norm,      # [2] - agent y, x position
        agent_dir_onehot,    # [4] - agent direction one-hot
        goal_pos_norm,       # [2] - goal y, x position  
        jnp.array([dist_norm]),  # [1] - distance to goal
        jnp.array([step_norm])   # [1] - step progress
    ])
    
    return obs.astype(jnp.float32)


def compute_reward(state, action, next_state):
    """Compute reward for navigation task.
    
    Reward components:
    1. Distance improvement: +1.0 for getting closer, -1.0 for getting farther
    2. Small step penalty: -0.01 to encourage efficiency
    3. Goal bonus: +10.0 for reaching the goal
    """
    H, W = state.grid.shape[:2]
    
    # Find goal position (same logic as observation)
    goal_mask = (state.grid[:, :, 0] == 6)
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    goal_count = jnp.maximum(jnp.sum(goal_mask), 1.0)
    goal_y = jnp.sum(yy * goal_mask) / goal_count
    goal_x = jnp.sum(xx * goal_mask) / goal_count
    
    # Calculate distances before and after action
    prev_pos = state.agent.position.astype(jnp.float32)
    curr_pos = next_state.agent.position.astype(jnp.float32)
    
    prev_dist = jnp.abs(prev_pos[0] - goal_y) + jnp.abs(prev_pos[1] - goal_x)
    curr_dist = jnp.abs(curr_pos[0] - goal_y) + jnp.abs(curr_pos[1] - goal_x)
    
    # Distance improvement reward
    dist_improvement = prev_dist - curr_dist
    distance_reward = dist_improvement * 1.0
    
    # Small step penalty to encourage efficiency
    step_penalty = -0.01
    
    # Check if agent is on goal tile
    agent_y, agent_x = curr_pos[0].astype(jnp.int32), curr_pos[1].astype(jnp.int32)
    on_goal = (next_state.grid[agent_y, agent_x, 0] == 6)
    goal_bonus = jax.lax.select(on_goal, 10.0, 0.0)
    
    # Total reward
    total_reward = distance_reward + step_penalty + goal_bonus
    
    return total_reward.astype(jnp.float32)
