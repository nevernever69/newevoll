"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Navigate to the goal tile in a 6x6 empty room.
Success rate: 44%
Obs dim:      20
Generation:   3
Iteration:    20
"""

import jax
import jax.numpy as jnp

def get_observation(state):
    """Extract focused observation for navigation task."""
    H, W = state.grid.shape[:2]
    
    # Agent position normalized to [0, 1]
    agent_pos = state.agent.position.astype(jnp.float32)
    norm_pos = agent_pos / jnp.array([H - 1, W - 1], dtype=jnp.float32)
    
    # Agent direction as one-hot encoding
    direction_onehot = jax.nn.one_hot(state.agent.direction, 4, dtype=jnp.float32)
    
    # Find goal positions
    goal_mask = (state.grid[:, :, 0] == 6)  # GOAL tile type
    
    # Create coordinate grids
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    # Compute goal center
    goal_count = jnp.sum(goal_mask)
    goal_y = jnp.sum(yy * goal_mask) / jnp.maximum(goal_count, 1.0)
    goal_x = jnp.sum(xx * goal_mask) / jnp.maximum(goal_count, 1.0)
    goal_center = jnp.array([goal_y, goal_x]) / jnp.array([H - 1, W - 1])
    
    # Raw goal delta (unnormalized for clearer signals)
    goal_delta = jnp.array([goal_y - agent_pos[0], goal_x - agent_pos[1]])
    
    # Manhattan distance to goal (normalized)
    manhattan_dist = jnp.abs(goal_delta[0]) + jnp.abs(goal_delta[1])
    max_dist = H + W - 2
    norm_manhattan = manhattan_dist / max_dist
    
    # Direction vectors: Up=(-1,0), Right=(0,1), Down=(1,0), Left=(0,-1)
    DIRECTION_VECTORS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.float32)
    
    # Compute optimal direction to goal (which cardinal direction minimizes distance)
    optimal_moves = jnp.array([
        jnp.maximum(-goal_delta[0], 0.0),  # up moves needed
        jnp.maximum(goal_delta[1], 0.0),   # right moves needed  
        jnp.maximum(goal_delta[0], 0.0),   # down moves needed
        jnp.maximum(-goal_delta[1], 0.0)   # left moves needed
    ])
    
    # Normalize by total moves needed
    total_moves = jnp.sum(optimal_moves)
    optimal_direction_probs = optimal_moves / jnp.maximum(total_moves, 1.0)
    
    # Current agent facing direction
    agent_facing = DIRECTION_VECTORS[state.agent.direction]
    
    # Alignment of current direction with goal delta
    goal_direction_norm = goal_delta / jnp.maximum(jnp.linalg.norm(goal_delta), 1e-8)
    current_alignment = jnp.dot(goal_direction_norm, agent_facing)
    
    # Compute alignment for each possible direction
    all_alignments = jnp.array([
        jnp.dot(goal_direction_norm, DIRECTION_VECTORS[0]),  # up
        jnp.dot(goal_direction_norm, DIRECTION_VECTORS[1]),  # right
        jnp.dot(goal_direction_norm, DIRECTION_VECTORS[2]),  # down
        jnp.dot(goal_direction_norm, DIRECTION_VECTORS[3])   # left
    ])
    
    # Best alignment possible
    best_alignment = jnp.max(all_alignments)
    
    # How many turns needed to face optimal direction
    best_direction = jnp.argmax(all_alignments)
    turns_needed = jnp.minimum(
        (best_direction - state.agent.direction) % 4,
        (state.agent.direction - best_direction) % 4
    )
    
    # Simple action guidance: value of each action
    # Forward value based on current alignment
    forward_value = jnp.maximum(current_alignment, 0.0)
    
    # Turn values based on alignment improvement
    left_dir = (state.agent.direction - 1) % 4
    right_dir = (state.agent.direction + 1) % 4
    left_alignment = all_alignments[left_dir]
    right_alignment = all_alignments[right_dir]
    
    left_turn_value = jnp.maximum(left_alignment - current_alignment, 0.0)
    right_turn_value = jnp.maximum(right_alignment - current_alignment, 0.0)
    
    # Step count normalized
    norm_step = state.step_num.astype(jnp.float32) / 100.0
    
    # Distance components (signed, showing which direction to go)
    delta_y_norm = goal_delta[0] / (H - 1)  # negative means go up, positive means go down
    delta_x_norm = goal_delta[1] / (W - 1)  # negative means go left, positive means go right
    
    # Proximity indicators
    very_close = (norm_manhattan < 0.05).astype(jnp.float32)
    close = (norm_manhattan < 0.15).astype(jnp.float32)
    
    # Concatenate all features (19 elements total)
    observation = jnp.concatenate([
        norm_pos,                                              # 2: agent position
        direction_onehot,                                      # 4: agent direction
        goal_center,                                           # 2: goal position
        jnp.array([delta_y_norm, delta_x_norm]),              # 2: signed direction to goal
        jnp.array([norm_manhattan]),                          # 1: distance to goal
        jnp.array([current_alignment, best_alignment]),       # 2: alignment metrics
        jnp.array([forward_value, left_turn_value, right_turn_value]), # 3: action values
        jnp.array([turns_needed.astype(jnp.float32) / 2.0]),  # 1: turns to optimal (normalized)
        jnp.array([very_close, close]),                       # 2: proximity flags
        jnp.array([norm_step])                                # 1: time step
    ])
    
    return observation.astype(jnp.float32)

def compute_reward(state, action, next_state):
    """Compute reward with clear, consistent signals."""
    H, W = state.grid.shape[:2]
    
    # Check if agent is on a goal tile in next state
    agent_pos = next_state.agent.position
    agent_y, agent_x = agent_pos[0], agent_pos[1]
    on_goal = (next_state.grid[agent_y, agent_x, 0] == 6)
    
    # Large goal reached bonus
    goal_reward = jax.lax.select(on_goal, 100.0, 0.0)
    
    # Find goal center
    goal_mask = (next_state.grid[:, :, 0] == 6)
    goal_count = jnp.sum(goal_mask)
    
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    goal_y = jnp.sum(yy * goal_mask) / jnp.maximum(goal_count, 1.0)
    goal_x = jnp.sum(xx * goal_mask) / jnp.maximum(goal_count, 1.0)
    
    # Manhattan distance progress
    curr_pos = next_state.agent.position.astype(jnp.float32)
    prev_pos = state.agent.position.astype(jnp.float32)
    
    curr_dist = jnp.abs(curr_pos[0] - goal_y) + jnp.abs(curr_pos[1] - goal_x)
    prev_dist = jnp.abs(prev_pos[0] - goal_y) + jnp.abs(prev_pos[1] - goal_x)
    
    # Strong progress reward - this is the main learning signal
    progress_reward = (prev_dist - curr_dist) * 5.0
    
    # Bonus for being close to goal (exponential)
    proximity_bonus = jnp.exp(-curr_dist / 3.0) * 0.5
    
    # Action-specific rewards
    goal_direction = jnp.array([goal_y - prev_pos[0], goal_x - prev_pos[1]])
    goal_direction_norm = goal_direction / jnp.maximum(jnp.linalg.norm(goal_direction), 1e-8)
    
    DIRECTION_VECTORS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.float32)
    agent_facing = DIRECTION_VECTORS[state.agent.direction]
    
    alignment = jnp.dot(goal_direction_norm, agent_facing)
    
    # Reward forward movement when reasonably aligned
    forward_action = (action == 0)
    forward_bonus = jax.lax.select(
        jnp.logical_and(forward_action, alignment > 0.2),
        alignment * 0.8,
        0.0
    )
    
    # Reward turns that improve alignment, penalize unnecessary turns
    turning = jnp.logical_or(action == 1, action == 2)
    
    # Compute new alignment after turn
    new_direction = jax.lax.select(
        action == 1, (state.agent.direction + 1) % 4,
        jax.lax.select(action == 2, (state.agent.direction - 1) % 4, state.agent.direction)
    )
    new_facing = DIRECTION_VECTORS[new_direction]
    new_alignment = jnp.dot(goal_direction_norm, new_facing)
    
    # Reward good turns, penalize bad ones
    turn_reward = jax.lax.select(
        turning,
        (new_alignment - alignment) * 1.0,  # Reward improvement, penalize degradation
        0.0
    )
    
    # Small time penalty to encourage efficiency
    time_penalty = -0.02
    
    # Total reward
    total_reward = goal_reward + progress_reward + proximity_bonus + forward_bonus + turn_reward + time_penalty
    
    return total_reward.astype(jnp.float32)
