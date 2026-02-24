"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Pick up the blue pyramid. Grid 9x9, max_steps=80.
Success rate: 98%
Obs dim:      190
Generation:   3
Iteration:    30
Mode:         obs_only
"""

import jax
import jax.numpy as jnp

def get_observation(state) -> jnp.ndarray:
    H, W = state.grid.shape[0], state.grid.shape[1]
    
    PYRAMID_ID = 5
    BLUE_ID = 3
    
    # Agent position normalized
    agent_y = state.agent.position[0].astype(jnp.float32) / (H - 1)
    agent_x = state.agent.position[1].astype(jnp.float32) / (W - 1)
    
    # Agent direction as one-hot (4 directions)
    dir_oh = jax.nn.one_hot(state.agent.direction, 4)
    
    # Pocket contents (raw + normalized)
    pocket_tile = state.agent.pocket[0].astype(jnp.float32) / 12.0
    pocket_color = state.agent.pocket[1].astype(jnp.float32) / 11.0
    
    # Whether holding blue pyramid
    holding_blue_pyramid = ((state.agent.pocket[0] == PYRAMID_ID) & (state.agent.pocket[1] == BLUE_ID)).astype(jnp.float32)
    pocket_empty = (state.agent.pocket[0] == 0).astype(jnp.float32)
    
    # Find blue pyramid location
    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    
    pyramid_mask = (state.grid[:, :, 0] == PYRAMID_ID) & (state.grid[:, :, 1] == BLUE_ID)
    pyramid_count = jnp.sum(pyramid_mask).astype(jnp.float32)
    pyramid_y = jnp.sum(yy * pyramid_mask).astype(jnp.float32) / jnp.maximum(pyramid_count, 1.0)
    pyramid_x = jnp.sum(xx * pyramid_mask).astype(jnp.float32) / jnp.maximum(pyramid_count, 1.0)
    
    # Whether pyramid exists on grid
    pyramid_exists = jnp.minimum(pyramid_count, 1.0)
    
    # Normalized absolute pyramid position
    pyramid_y_norm = pyramid_y / (H - 1)
    pyramid_x_norm = pyramid_x / (W - 1)
    
    # Raw relative position
    raw_rel_y = pyramid_y - state.agent.position[0].astype(jnp.float32)
    raw_rel_x = pyramid_x - state.agent.position[1].astype(jnp.float32)
    
    # Normalized relative position
    rel_y = raw_rel_y / (H - 1)
    rel_x = raw_rel_x / (W - 1)
    
    # Manhattan distance to pyramid (normalized)
    manhattan_dist = (jnp.abs(raw_rel_y) + jnp.abs(raw_rel_x)) / (H + W - 2)
    
    # Euclidean distance (normalized)
    dist_raw = jnp.sqrt(raw_rel_y**2 + raw_rel_x**2 + 1e-8)
    dist_norm = dist_raw / jnp.sqrt((H-1)**2 + (W-1)**2 + 1e-8)
    
    # Angle to pyramid (encoded as sin/cos), zeroed when pyramid not on grid
    angle_sin = (raw_rel_y / dist_raw) * pyramid_exists
    angle_cos = (raw_rel_x / dist_raw) * pyramid_exists
    
    # Direction vectors
    DIRECTIONS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)
    dy = DIRECTIONS[state.agent.direction, 0]
    dx = DIRECTIONS[state.agent.direction, 1]
    
    # Front tile
    front_y = jnp.clip(state.agent.position[0].astype(jnp.int32) + dy, 0, H - 1)
    front_x = jnp.clip(state.agent.position[1].astype(jnp.int32) + dx, 0, W - 1)
    front_tile_id = state.grid[front_y, front_x, 0].astype(jnp.float32) / 12.0
    front_tile_color = state.grid[front_y, front_x, 1].astype(jnp.float32) / 11.0
    
    # Is front tile the blue pyramid?
    front_is_blue_pyramid = ((state.grid[front_y, front_x, 0] == PYRAMID_ID) & 
                              (state.grid[front_y, front_x, 1] == BLUE_ID)).astype(jnp.float32)
    
    # Derived features
    adjacent_to_pyramid = ((jnp.abs(raw_rel_y) + jnp.abs(raw_rel_x)) <= 1.5).astype(jnp.float32)
    
    facing_dy = DIRECTIONS[state.agent.direction, 0].astype(jnp.float32)
    facing_dx = DIRECTIONS[state.agent.direction, 1].astype(jnp.float32)
    dot_product = facing_dy * raw_rel_y + facing_dx * raw_rel_x
    facing_toward_pyramid = (dot_product > 0).astype(jnp.float32)
    
    same_row = (jnp.abs(raw_rel_y) < 0.5).astype(jnp.float32)
    same_col = (jnp.abs(raw_rel_x) < 0.5).astype(jnp.float32)
    
    # Sign direction to pyramid
    dy_to_pyramid = jnp.sign(raw_rel_y)
    dx_to_pyramid = jnp.sign(raw_rel_x)
    
    can_pickup = front_is_blue_pyramid * pocket_empty
    task_done = holding_blue_pyramid
    
    # Step progress
    step_norm = state.step_num.astype(jnp.float32) / 80.0
    
    # All 4 neighbors of the agent (up, right, down, left)
    neighbor_features = []
    for d in range(4):
        ndy = DIRECTIONS[d, 0]
        ndx = DIRECTIONS[d, 1]
        ny = jnp.clip(state.agent.position[0].astype(jnp.int32) + ndy, 0, H - 1)
        nx = jnp.clip(state.agent.position[1].astype(jnp.int32) + ndx, 0, W - 1)
        t_id = state.grid[ny, nx, 0].astype(jnp.float32) / 12.0
        t_col = state.grid[ny, nx, 1].astype(jnp.float32) / 11.0
        is_bp = ((state.grid[ny, nx, 0] == PYRAMID_ID) & (state.grid[ny, nx, 1] == BLUE_ID)).astype(jnp.float32)
        neighbor_features.extend([t_id, t_col, is_bp])
    neighbors = jnp.array(neighbor_features)  # 4*3 = 12
    
    # Local view: 7x7 grid around agent (tile_id, color_id, is_blue_pyramid)
    local_tiles = []
    for dy_off in [-3, -2, -1, 0, 1, 2, 3]:
        for dx_off in [-3, -2, -1, 0, 1, 2, 3]:
            ny = jnp.clip(state.agent.position[0].astype(jnp.int32) + dy_off, 0, H - 1)
            nx = jnp.clip(state.agent.position[1].astype(jnp.int32) + dx_off, 0, W - 1)
            t_id = state.grid[ny, nx, 0].astype(jnp.float32) / 12.0
            t_col = state.grid[ny, nx, 1].astype(jnp.float32) / 11.0
            is_bp = ((state.grid[ny, nx, 0] == PYRAMID_ID) & (state.grid[ny, nx, 1] == BLUE_ID)).astype(jnp.float32)
            local_tiles.extend([t_id, t_col, is_bp])
    local_view = jnp.array(local_tiles)  # 7*7*3 = 147
    
    # Assemble observation
    obs = jnp.concatenate([
        jnp.array([agent_y, agent_x]),                                          # 2
        dir_oh,                                                                   # 4
        jnp.array([pocket_tile, pocket_color]),                                   # 2
        jnp.array([holding_blue_pyramid, pocket_empty]),                          # 2
        jnp.array([pyramid_y_norm, pyramid_x_norm, pyramid_exists]),              # 3
        jnp.array([rel_y, rel_x, manhattan_dist, dist_norm]),                     # 4
        jnp.array([angle_sin, angle_cos]),                                        # 2
        jnp.array([front_tile_id, front_tile_color, front_is_blue_pyramid]),      # 3
        jnp.array([step_norm]),                                                   # 1
        jnp.array([dy_to_pyramid, dx_to_pyramid]),                               # 2
        jnp.array([adjacent_to_pyramid, facing_toward_pyramid]),                  # 2
        jnp.array([same_row, same_col]),                                          # 2
        jnp.array([can_pickup, task_done]),                                       # 2
        neighbors,                                                                # 12
        local_view,                                                               # 147
    ])
    
    return obs.astype(jnp.float32)
