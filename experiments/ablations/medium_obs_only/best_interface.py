"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Place the yellow pyramid adjacent to the green square. Grid 9x9, max_steps=80.
Success rate: 99%
Obs dim:      108
Generation:   2
Iteration:    30
Mode:         obs_only
"""

import jax
import jax.numpy as jnp

def get_observation(state) -> jnp.ndarray:
    H, W = state.grid.shape[:2]
    
    DIRECTIONS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.float32)
    
    # Agent info
    agent_y = state.agent.position[0].astype(jnp.float32)
    agent_x = state.agent.position[1].astype(jnp.float32)
    norm_agent_y = agent_y / (H - 1)
    norm_agent_x = agent_x / (W - 1)
    dir_one_hot = jax.nn.one_hot(state.agent.direction, 4)
    
    # Pocket info
    pocket_tile = state.agent.pocket[0].astype(jnp.float32) / 12.0
    pocket_color = state.agent.pocket[1].astype(jnp.float32) / 11.0
    holding_yellow_pyramid = ((state.agent.pocket[0] == 5) & (state.agent.pocket[1] == 5)).astype(jnp.float32)
    pocket_empty = (state.agent.pocket[0] == 0).astype(jnp.float32)
    
    # Grid coordinates
    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    yy_f = yy.astype(jnp.float32)
    xx_f = xx.astype(jnp.float32)
    
    # Find yellow pyramid (tile=5, color=5)
    pyr_mask = (state.grid[:, :, 0] == 5) & (state.grid[:, :, 1] == 5)
    pyr_count = jnp.sum(pyr_mask).astype(jnp.float32)
    pyr_y = jnp.sum(yy_f * pyr_mask) / jnp.maximum(pyr_count, 1.0)
    pyr_x = jnp.sum(xx_f * pyr_mask) / jnp.maximum(pyr_count, 1.0)
    pyr_found = jnp.minimum(pyr_count, 1.0)
    
    # Find green square (tile=4, color=2)
    sq_mask = (state.grid[:, :, 0] == 4) & (state.grid[:, :, 1] == 2)
    sq_count = jnp.sum(sq_mask).astype(jnp.float32)
    sq_y = jnp.sum(yy_f * sq_mask) / jnp.maximum(sq_count, 1.0)
    sq_x = jnp.sum(xx_f * sq_mask) / jnp.maximum(sq_count, 1.0)
    sq_found = jnp.minimum(sq_count, 1.0)
    
    # Effective pyramid position (agent pos if held)
    eff_pyr_y = jnp.where(holding_yellow_pyramid > 0, agent_y, pyr_y)
    eff_pyr_x = jnp.where(holding_yellow_pyramid > 0, agent_x, pyr_x)
    
    # Normalized positions
    norm_pyr_y = pyr_y / (H - 1)
    norm_pyr_x = pyr_x / (W - 1)
    norm_sq_y = sq_y / (H - 1)
    norm_sq_x = sq_x / (W - 1)
    norm_eff_pyr_y = eff_pyr_y / (H - 1)
    norm_eff_pyr_x = eff_pyr_x / (W - 1)
    
    # Relative vectors: agent -> pyramid (on grid)
    rel_pyr_y = (pyr_y - agent_y) / (H - 1)
    rel_pyr_x = (pyr_x - agent_x) / (W - 1)
    dist_to_pyr = (jnp.abs(pyr_y - agent_y) + jnp.abs(pyr_x - agent_x)) / (H + W - 2)
    
    # Relative vectors: agent -> square
    rel_sq_y = (sq_y - agent_y) / (H - 1)
    rel_sq_x = (sq_x - agent_x) / (W - 1)
    dist_to_sq = (jnp.abs(sq_y - agent_y) + jnp.abs(sq_x - agent_x)) / (H + W - 2)
    
    # Relative vectors: effective pyramid -> square
    rel_eff_pyr_to_sq_y = (sq_y - eff_pyr_y) / (H - 1)
    rel_eff_pyr_to_sq_x = (sq_x - eff_pyr_x) / (W - 1)
    dist_eff_pyr_to_sq = (jnp.abs(sq_y - eff_pyr_y) + jnp.abs(sq_x - eff_pyr_x)) / (H + W - 2)
    
    # Pyramid on grid to square distance
    dist_pyr_to_sq = (jnp.abs(sq_y - pyr_y) + jnp.abs(sq_x - pyr_x)) / (H + W - 2)
    
    # Is pyramid adjacent to green square?
    pyr_adj_to_sq = (jnp.abs(pyr_y - sq_y) + jnp.abs(pyr_x - sq_x) <= 1.0).astype(jnp.float32) * pyr_found
    
    # Front tile info
    dy = DIRECTIONS[state.agent.direction, 0]
    dx = DIRECTIONS[state.agent.direction, 1]
    front_y = jnp.clip(agent_y + dy, 0, H - 1).astype(jnp.int32)
    front_x = jnp.clip(agent_x + dx, 0, W - 1).astype(jnp.int32)
    front_tile_id = state.grid[front_y, front_x, 0].astype(jnp.float32) / 12.0
    front_tile_color = state.grid[front_y, front_x, 1].astype(jnp.float32) / 11.0
    front_is_pyr = ((state.grid[front_y, front_x, 0] == 5) & (state.grid[front_y, front_x, 1] == 5)).astype(jnp.float32)
    front_is_sq = ((state.grid[front_y, front_x, 0] == 4) & (state.grid[front_y, front_x, 1] == 2)).astype(jnp.float32)
    front_adj_to_sq = ((jnp.abs(front_y - sq_y) + jnp.abs(front_x - sq_x)) <= 1.0).astype(jnp.float32)
    front_is_floor = (state.grid[front_y, front_x, 0] == 1).astype(jnp.float32)
    front_good_putdown = front_is_floor * front_adj_to_sq
    
    # Agent adjacency to square
    agent_adj_to_sq = ((jnp.abs(agent_y - sq_y) + jnp.abs(agent_x - sq_x)) <= 1.0).astype(jnp.float32)
    
    # Facing toward pyramid
    dir_dy = DIRECTIONS[state.agent.direction, 0]
    dir_dx = DIRECTIONS[state.agent.direction, 1]
    pyr_dy_sign = jnp.sign(pyr_y - agent_y)
    pyr_dx_sign = jnp.sign(pyr_x - agent_x)
    facing_pyr = jnp.clip((pyr_dy_sign * dir_dy + pyr_dx_sign * dir_dx) / 2.0, -1.0, 1.0)
    
    # Facing toward square
    sq_dy_sign = jnp.sign(sq_y - agent_y)
    sq_dx_sign = jnp.sign(sq_x - agent_x)
    facing_sq = jnp.clip((sq_dy_sign * dir_dy + sq_dx_sign * dir_dx) / 2.0, -1.0, 1.0)
    
    # 4 neighbors of green square
    sq_y_int = sq_y.astype(jnp.int32)
    sq_x_int = sq_x.astype(jnp.int32)
    
    n_up_y = jnp.clip(sq_y_int - 1, 0, H - 1)
    n_dn_y = jnp.clip(sq_y_int + 1, 0, H - 1)
    n_lt_x = jnp.clip(sq_x_int - 1, 0, W - 1)
    n_rt_x = jnp.clip(sq_x_int + 1, 0, W - 1)
    
    # Check if neighbors are walkable floor
    n_up_floor = (state.grid[n_up_y, sq_x_int, 0] == 1).astype(jnp.float32)
    n_dn_floor = (state.grid[n_dn_y, sq_x_int, 0] == 1).astype(jnp.float32)
    n_lt_floor = (state.grid[sq_y_int, n_lt_x, 0] == 1).astype(jnp.float32)
    n_rt_floor = (state.grid[sq_y_int, n_rt_x, 0] == 1).astype(jnp.float32)
    
    # Normalized neighbor positions
    norm_n_up_y = n_up_y.astype(jnp.float32) / (H - 1)
    norm_n_dn_y = n_dn_y.astype(jnp.float32) / (H - 1)
    norm_n_lt_x = n_lt_x.astype(jnp.float32) / (W - 1)
    norm_n_rt_x = n_rt_x.astype(jnp.float32) / (W - 1)
    
    # Distance from agent to each valid neighbor of square
    dist_to_n_up = (jnp.abs(agent_y - n_up_y) + jnp.abs(agent_x - sq_x_int)) / (H + W - 2)
    dist_to_n_dn = (jnp.abs(agent_y - n_dn_y) + jnp.abs(agent_x - sq_x_int)) / (H + W - 2)
    dist_to_n_lt = (jnp.abs(agent_y - sq_y_int) + jnp.abs(agent_x - n_lt_x)) / (H + W - 2)
    dist_to_n_rt = (jnp.abs(agent_y - sq_y_int) + jnp.abs(agent_x - n_rt_x)) / (H + W - 2)
    
    # Best (closest valid floor) neighbor distance
    best_n_dist = jnp.minimum(
        jnp.minimum(
            jnp.where(n_up_floor > 0, dist_to_n_up, 1.0),
            jnp.where(n_dn_floor > 0, dist_to_n_dn, 1.0)
        ),
        jnp.minimum(
            jnp.where(n_lt_floor > 0, dist_to_n_lt, 1.0),
            jnp.where(n_rt_floor > 0, dist_to_n_rt, 1.0)
        )
    )
    
    # Best neighbor position (closest valid floor neighbor)
    best_n_y = jnp.where(
        jnp.where(n_up_floor > 0, dist_to_n_up, 1.0) <= jnp.where(n_dn_floor > 0, dist_to_n_dn, 1.0),
        jnp.where(n_up_floor > 0, n_up_y.astype(jnp.float32), n_dn_y.astype(jnp.float32)),
        jnp.where(n_dn_floor > 0, n_dn_y.astype(jnp.float32), n_up_y.astype(jnp.float32))
    )
    best_n_x = jnp.where(
        jnp.where(n_lt_floor > 0, dist_to_n_lt, 1.0) <= jnp.where(n_rt_floor > 0, dist_to_n_rt, 1.0),
        jnp.where(n_lt_floor > 0, n_lt_x.astype(jnp.float32), n_rt_x.astype(jnp.float32)),
        jnp.where(n_rt_floor > 0, n_rt_x.astype(jnp.float32), n_lt_x.astype(jnp.float32))
    )
    
    # Relative: agent -> best neighbor
    rel_to_best_n_y = (best_n_y - agent_y) / (H - 1)
    rel_to_best_n_x = (best_n_x - agent_x) / (W - 1)
    
    # Task phase indicators
    phase_pickup = ((holding_yellow_pyramid < 0.5) & (pyr_found > 0.5)).astype(jnp.float32)
    phase_place = holding_yellow_pyramid
    phase_done = pyr_adj_to_sq
    
    # Step progress
    step_norm = state.step_num.astype(jnp.float32) / 80.0
    
    # Local 5x5 view around agent
    view_radius = 2
    agent_y_int = state.agent.position[0].astype(jnp.int32)
    agent_x_int = state.agent.position[1].astype(jnp.int32)
    
    offsets = jnp.arange(-view_radius, view_radius + 1)
    oy, ox = jnp.meshgrid(offsets, offsets, indexing='ij')
    
    view_ys = jnp.clip(agent_y_int + oy, 0, H - 1)
    view_xs = jnp.clip(agent_x_int + ox, 0, W - 1)
    
    view_tiles = state.grid[view_ys, view_xs, 0].astype(jnp.float32) / 12.0
    view_colors = state.grid[view_ys, view_xs, 1].astype(jnp.float32) / 11.0
    
    local_tiles = view_tiles.reshape(-1)   # 25
    local_colors = view_colors.reshape(-1) # 25
    
    obs = jnp.concatenate([
        jnp.array([
            norm_agent_y, norm_agent_x,                      # 2
            norm_pyr_y, norm_pyr_x,                          # 2
            norm_eff_pyr_y, norm_eff_pyr_x,                  # 2
            norm_sq_y, norm_sq_x,                            # 2
            rel_pyr_y, rel_pyr_x,                            # 2
            rel_sq_y, rel_sq_x,                              # 2
            rel_eff_pyr_to_sq_y, rel_eff_pyr_to_sq_x,       # 2
            dist_to_pyr,                                      # 1
            dist_to_sq,                                       # 1
            dist_pyr_to_sq,                                   # 1
            dist_eff_pyr_to_sq,                               # 1
            pyr_found,                                        # 1
            sq_found,                                         # 1
            holding_yellow_pyramid,                           # 1
            pocket_empty,                                     # 1
            pocket_tile, pocket_color,                        # 2
            front_tile_id, front_tile_color,                  # 2
            front_is_pyr,                                     # 1
            front_is_sq,                                      # 1
            front_adj_to_sq,                                  # 1
            front_is_floor,                                   # 1
            front_good_putdown,                               # 1
            agent_adj_to_sq,                                  # 1
            pyr_adj_to_sq,                                    # 1
            facing_pyr,                                       # 1
            facing_sq,                                        # 1
            best_n_dist,                                      # 1
            rel_to_best_n_y, rel_to_best_n_x,               # 2
            norm_n_up_y, norm_n_dn_y,                        # 2
            norm_n_lt_x, norm_n_rt_x,                        # 2
            n_up_floor, n_dn_floor, n_lt_floor, n_rt_floor,  # 4
            dist_to_n_up, dist_to_n_dn,                      # 2
            dist_to_n_lt, dist_to_n_rt,                      # 2
            phase_pickup, phase_place, phase_done,            # 3
            step_norm,                                        # 1
        ], dtype=jnp.float32),
        dir_one_hot,   # 4
        local_tiles,   # 25
        local_colors,  # 25
    ])
    
    return obs.astype(jnp.float32)
