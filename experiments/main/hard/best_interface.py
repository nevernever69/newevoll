"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Pick up blue pyramid (transforms to green ball in hand), then place green ball next to yellow hex. 4-room 13x13 grid with distractors, max_steps=400.
Success rate: 76%
Obs dim:      147
Generation:   2
Iteration:    30
Mode:         full
"""

import jax
import jax.numpy as jnp

# Tile and color constants
FLOOR = 1
WALL = 2
BALL = 3
PYRAMID = 5
HEX = 11
EMPTY = 0

GREEN = 2
BLUE = 3
YELLOW = 5

DIRECTIONS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)


def get_observation(state):
    grid = state.grid
    H, W = grid.shape[:2]
    H_f = jnp.float32(H)
    W_f = jnp.float32(W)

    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    agent_y = state.agent.position[0].astype(jnp.float32)
    agent_x = state.agent.position[1].astype(jnp.float32)
    norm_y = agent_y / (H_f - 1)
    norm_x = agent_x / (W_f - 1)
    dir_oh = jax.nn.one_hot(state.agent.direction, 4)

    # Pocket info
    holding_green_ball = ((state.agent.pocket[0] == BALL) & (state.agent.pocket[1] == GREEN)).astype(jnp.float32)
    pocket_empty = (state.agent.pocket[0] == EMPTY).astype(jnp.float32)
    pocket_tile_norm = state.agent.pocket[0].astype(jnp.float32) / 12.0
    pocket_color_norm = state.agent.pocket[1].astype(jnp.float32) / 11.0

    # Blue pyramid location
    blue_pyr_mask = (grid[:, :, 0] == PYRAMID) & (grid[:, :, 1] == BLUE)
    blue_pyr_count = jnp.sum(blue_pyr_mask).astype(jnp.float32)
    blue_pyr_y = jnp.sum(yy * blue_pyr_mask).astype(jnp.float32) / jnp.maximum(blue_pyr_count, 1.0)
    blue_pyr_x = jnp.sum(xx * blue_pyr_mask).astype(jnp.float32) / jnp.maximum(blue_pyr_count, 1.0)
    blue_pyr_norm_y = blue_pyr_y / (H_f - 1)
    blue_pyr_norm_x = blue_pyr_x / (W_f - 1)
    blue_pyr_exists = jnp.minimum(blue_pyr_count, 1.0)
    rel_pyr_y = (blue_pyr_y - agent_y) / (H_f - 1)
    rel_pyr_x = (blue_pyr_x - agent_x) / (W_f - 1)
    dist_to_pyr = (jnp.abs(blue_pyr_y - agent_y) + jnp.abs(blue_pyr_x - agent_x)) / (H_f + W_f - 2)

    # Yellow hex location
    yellow_hex_mask = (grid[:, :, 0] == HEX) & (grid[:, :, 1] == YELLOW)
    yellow_hex_count = jnp.sum(yellow_hex_mask).astype(jnp.float32)
    yellow_hex_y = jnp.sum(yy * yellow_hex_mask).astype(jnp.float32) / jnp.maximum(yellow_hex_count, 1.0)
    yellow_hex_x = jnp.sum(xx * yellow_hex_mask).astype(jnp.float32) / jnp.maximum(yellow_hex_count, 1.0)
    yellow_hex_norm_y = yellow_hex_y / (H_f - 1)
    yellow_hex_norm_x = yellow_hex_x / (W_f - 1)
    yellow_hex_exists = jnp.minimum(yellow_hex_count, 1.0)
    rel_hex_y = (yellow_hex_y - agent_y) / (H_f - 1)
    rel_hex_x = (yellow_hex_x - agent_x) / (W_f - 1)
    dist_to_hex = (jnp.abs(yellow_hex_y - agent_y) + jnp.abs(yellow_hex_x - agent_x)) / (H_f + W_f - 2)

    hex_y_int = yellow_hex_y.astype(jnp.int32)
    hex_x_int = yellow_hex_x.astype(jnp.int32)

    # Front tile info
    dy = DIRECTIONS[state.agent.direction][0]
    dx = DIRECTIONS[state.agent.direction][1]
    front_y = jnp.clip(state.agent.position[0].astype(jnp.int32) + dy, 0, H - 1)
    front_x = jnp.clip(state.agent.position[1].astype(jnp.int32) + dx, 0, W - 1)
    front_tile_id = grid[front_y, front_x, 0].astype(jnp.float32) / 12.0
    front_tile_color = grid[front_y, front_x, 1].astype(jnp.float32) / 11.0
    front_is_blue_pyr = ((grid[front_y, front_x, 0] == PYRAMID) & (grid[front_y, front_x, 1] == BLUE)).astype(jnp.float32)
    front_is_floor = (grid[front_y, front_x, 0] == FLOOR).astype(jnp.float32)
    front_is_yellow_hex = ((grid[front_y, front_x, 0] == HEX) & (grid[front_y, front_x, 1] == YELLOW)).astype(jnp.float32)
    front_adj_to_hex = (jnp.abs(jnp.float32(front_y) - yellow_hex_y) + jnp.abs(jnp.float32(front_x) - yellow_hex_x) <= 1.0).astype(jnp.float32)
    can_putdown_here = (front_is_floor * front_adj_to_hex)
    ready_to_putdown_now = (holding_green_ball * can_putdown_here)

    # 4 neighbors of agent
    n_features = []
    for d in range(4):
        ny = jnp.clip(state.agent.position[0].astype(jnp.int32) + DIRECTIONS[d][0], 0, H - 1)
        nx = jnp.clip(state.agent.position[1].astype(jnp.int32) + DIRECTIONS[d][1], 0, W - 1)
        n_tile = grid[ny, nx, 0].astype(jnp.float32) / 12.0
        n_color = grid[ny, nx, 1].astype(jnp.float32) / 11.0
        n_is_pyr = ((grid[ny, nx, 0] == PYRAMID) & (grid[ny, nx, 1] == BLUE)).astype(jnp.float32)
        n_is_hex = ((grid[ny, nx, 0] == HEX) & (grid[ny, nx, 1] == YELLOW)).astype(jnp.float32)
        n_is_floor = (grid[ny, nx, 0] == FLOOR).astype(jnp.float32)
        n_adj_hex = (jnp.abs(jnp.float32(ny) - yellow_hex_y) + jnp.abs(jnp.float32(nx) - yellow_hex_x) <= 1.0).astype(jnp.float32)
        n_features.extend([n_tile, n_color, n_is_pyr, n_is_hex, n_is_floor, n_adj_hex])

    # Hex neighbors info
    hex_n_features = []
    for d in range(4):
        hny = jnp.clip(hex_y_int + DIRECTIONS[d][0], 0, H - 1)
        hnx = jnp.clip(hex_x_int + DIRECTIONS[d][1], 0, W - 1)
        hn_is_floor = (grid[hny, hnx, 0] == FLOOR).astype(jnp.float32)
        hn_rel_y = (jnp.float32(hny) - agent_y) / (H_f - 1)
        hn_rel_x = (jnp.float32(hnx) - agent_x) / (W_f - 1)
        hn_dist = (jnp.abs(jnp.float32(hny) - agent_y) + jnp.abs(jnp.float32(hnx) - agent_x)) / (H_f + W_f - 2)
        at_this_neighbor = ((jnp.abs(agent_y - jnp.float32(hny)) + jnp.abs(agent_x - jnp.float32(hnx))) < 0.5).astype(jnp.float32)
        hex_n_features.extend([hn_is_floor, hn_rel_y, hn_rel_x, hn_dist, at_this_neighbor])

    # Best placement: closest floor tile adjacent to hex
    best_place_y = jnp.float32(0)
    best_place_x = jnp.float32(0)
    best_dist = jnp.float32(1000.0)
    for d in range(4):
        hny = jnp.clip(hex_y_int + DIRECTIONS[d][0], 0, H - 1)
        hnx = jnp.clip(hex_x_int + DIRECTIONS[d][1], 0, W - 1)
        is_floor = (grid[hny, hnx, 0] == FLOOR)
        dist_d = jnp.abs(agent_y - jnp.float32(hny)) + jnp.abs(agent_x - jnp.float32(hnx))
        use_this = is_floor & (dist_d < best_dist)
        best_place_y = jax.lax.select(use_this, jnp.float32(hny), best_place_y)
        best_place_x = jax.lax.select(use_this, jnp.float32(hnx), best_place_x)
        best_dist = jax.lax.select(use_this, dist_d, best_dist)

    best_place_rel_y = (best_place_y - agent_y) / (H_f - 1)
    best_place_rel_x = (best_place_x - agent_x) / (W_f - 1)
    dist_to_best_place = (jnp.abs(agent_y - best_place_y) + jnp.abs(agent_x - best_place_x)) / (H_f + W_f - 2)
    at_best_place = (dist_to_best_place < 0.5 / (H_f + W_f - 2)).astype(jnp.float32)

    # Agent adjacency to hex
    agent_adj_to_hex = (jnp.abs(agent_y - yellow_hex_y) + jnp.abs(agent_x - yellow_hex_x) <= 1.0).astype(jnp.float32)

    # Green ball on ground
    green_ball_mask = (grid[:, :, 0] == BALL) & (grid[:, :, 1] == GREEN)
    green_ball_count = jnp.sum(green_ball_mask).astype(jnp.float32)
    green_ball_y = jnp.sum(yy * green_ball_mask).astype(jnp.float32) / jnp.maximum(green_ball_count, 1.0)
    green_ball_x = jnp.sum(xx * green_ball_mask).astype(jnp.float32) / jnp.maximum(green_ball_count, 1.0)
    green_ball_exists = jnp.minimum(green_ball_count, 1.0)
    dist_ball_to_hex = (jnp.abs(green_ball_y - yellow_hex_y) + jnp.abs(green_ball_x - yellow_hex_x)) / (H_f + W_f - 2)
    ball_adj_to_hex = ((jnp.abs(green_ball_y - yellow_hex_y) + jnp.abs(green_ball_x - yellow_hex_x)) <= 1.0).astype(jnp.float32) * green_ball_exists

    # Phase encoding
    phase0 = ((~(state.agent.pocket[0] == BALL)) & (blue_pyr_count > 0)).astype(jnp.float32)
    phase1 = holding_green_ball
    phase2 = ball_adj_to_hex

    # Step progress
    step_norm = state.step_num.astype(jnp.float32) / 400.0

    # Phase-dependent target
    target_y = jax.lax.select(holding_green_ball > 0.5, yellow_hex_y, blue_pyr_y)
    target_x = jax.lax.select(holding_green_ball > 0.5, yellow_hex_x, blue_pyr_x)
    rel_target_y = (target_y - agent_y) / (H_f - 1)
    rel_target_x = (target_x - agent_x) / (W_f - 1)
    dist_to_target = (jnp.abs(target_y - agent_y) + jnp.abs(target_x - agent_x)) / (H_f + W_f - 2)

    # Local 5x5 grid view
    local_view = []
    ay_int = state.agent.position[0].astype(jnp.int32)
    ax_int = state.agent.position[1].astype(jnp.int32)
    for dy_off in range(-2, 3):
        for dx_off in range(-2, 3):
            cy = jnp.clip(ay_int + dy_off, 0, H - 1)
            cx = jnp.clip(ax_int + dx_off, 0, W - 1)
            t = grid[cy, cx, 0].astype(jnp.float32) / 12.0
            c = grid[cy, cx, 1].astype(jnp.float32) / 11.0
            local_view.extend([t, c])

    # For each of 4 directions: is that tile a valid putdown spot?
    putdown_valid = []
    for d in range(4):
        ny = jnp.clip(state.agent.position[0].astype(jnp.int32) + DIRECTIONS[d][0], 0, H - 1)
        nx = jnp.clip(state.agent.position[1].astype(jnp.int32) + DIRECTIONS[d][1], 0, W - 1)
        is_floor = (grid[ny, nx, 0] == FLOOR).astype(jnp.float32)
        adj_hex = (jnp.abs(jnp.float32(ny) - yellow_hex_y) + jnp.abs(jnp.float32(nx) - yellow_hex_x) <= 1.0).astype(jnp.float32)
        is_facing = (state.agent.direction == d).astype(jnp.float32)
        putdown_valid.extend([is_floor * adj_hex, is_facing * is_floor * adj_hex])

    obs = jnp.concatenate([
        jnp.array([norm_y, norm_x], dtype=jnp.float32),                          # 2
        dir_oh.astype(jnp.float32),                                                # 4
        jnp.array([pocket_tile_norm, pocket_color_norm], dtype=jnp.float32),       # 2
        jnp.array([holding_green_ball, pocket_empty], dtype=jnp.float32),          # 2
        jnp.array([blue_pyr_norm_y, blue_pyr_norm_x, blue_pyr_exists], dtype=jnp.float32),  # 3
        jnp.array([rel_pyr_y, rel_pyr_x, dist_to_pyr], dtype=jnp.float32),        # 3
        jnp.array([yellow_hex_norm_y, yellow_hex_norm_x, yellow_hex_exists], dtype=jnp.float32), # 3
        jnp.array([rel_hex_y, rel_hex_x, dist_to_hex], dtype=jnp.float32),        # 3
        jnp.array([agent_adj_to_hex], dtype=jnp.float32),                          # 1
        jnp.array([front_tile_id, front_tile_color], dtype=jnp.float32),           # 2
        jnp.array([front_is_blue_pyr, front_is_floor, front_is_yellow_hex], dtype=jnp.float32), # 3
        jnp.array([front_adj_to_hex, can_putdown_here, ready_to_putdown_now], dtype=jnp.float32), # 3
        jnp.array(n_features, dtype=jnp.float32),                                  # 24
        jnp.array(hex_n_features, dtype=jnp.float32),                              # 20
        jnp.array([best_place_rel_y, best_place_rel_x, dist_to_best_place, at_best_place], dtype=jnp.float32), # 4
        jnp.array([green_ball_exists, dist_ball_to_hex, ball_adj_to_hex], dtype=jnp.float32), # 3
        jnp.array([phase0, phase1, phase2], dtype=jnp.float32),                    # 3
        jnp.array([rel_target_y, rel_target_x, dist_to_target], dtype=jnp.float32), # 3
        jnp.array([step_norm], dtype=jnp.float32),                                 # 1
        jnp.array(local_view, dtype=jnp.float32),                                  # 50
        jnp.array(putdown_valid, dtype=jnp.float32),                               # 8
    ])
    return obs.astype(jnp.float32)


def compute_reward(state, action, next_state):
    grid = state.grid
    next_grid = next_state.grid
    H, W = grid.shape[:2]
    H_f = jnp.float32(H)
    W_f = jnp.float32(W)
    norm_denom = H_f + W_f - 2.0

    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    # Pocket states
    was_holding_green_ball = (state.agent.pocket[0] == BALL) & (state.agent.pocket[1] == GREEN)
    now_holding_green_ball = (next_state.agent.pocket[0] == BALL) & (next_state.agent.pocket[1] == GREEN)

    # Agent positions
    agent_y = state.agent.position[0].astype(jnp.float32)
    agent_x = state.agent.position[1].astype(jnp.float32)
    next_agent_y = next_state.agent.position[0].astype(jnp.float32)
    next_agent_x = next_state.agent.position[1].astype(jnp.float32)

    # Blue pyramid location
    blue_pyr_mask = (grid[:, :, 0] == PYRAMID) & (grid[:, :, 1] == BLUE)
    blue_pyr_count = jnp.sum(blue_pyr_mask).astype(jnp.float32)
    pyr_y = jnp.sum(yy * blue_pyr_mask).astype(jnp.float32) / jnp.maximum(blue_pyr_count, 1.0)
    pyr_x = jnp.sum(xx * blue_pyr_mask).astype(jnp.float32) / jnp.maximum(blue_pyr_count, 1.0)

    # Yellow hex location
    yellow_hex_mask = (grid[:, :, 0] == HEX) & (grid[:, :, 1] == YELLOW)
    yellow_hex_count = jnp.sum(yellow_hex_mask).astype(jnp.float32)
    hex_y = jnp.sum(yy * yellow_hex_mask).astype(jnp.float32) / jnp.maximum(yellow_hex_count, 1.0)
    hex_x = jnp.sum(xx * yellow_hex_mask).astype(jnp.float32) / jnp.maximum(yellow_hex_count, 1.0)
    hex_y_int = hex_y.astype(jnp.int32)
    hex_x_int = hex_x.astype(jnp.int32)

    # Best placement position from current agent pos
    best_place_y = jnp.float32(0)
    best_place_x = jnp.float32(0)
    best_dist_val = jnp.float32(1000.0)
    for d in range(4):
        hny = jnp.clip(hex_y_int + DIRECTIONS[d][0], 0, H - 1)
        hnx = jnp.clip(hex_x_int + DIRECTIONS[d][1], 0, W - 1)
        is_floor = (grid[hny, hnx, 0] == FLOOR)
        dist_d = jnp.abs(agent_y - jnp.float32(hny)) + jnp.abs(agent_x - jnp.float32(hnx))
        use_this = is_floor & (dist_d < best_dist_val)
        best_place_y = jax.lax.select(use_this, jnp.float32(hny), best_place_y)
        best_place_x = jax.lax.select(use_this, jnp.float32(hnx), best_place_x)
        best_dist_val = jax.lax.select(use_this, dist_d, best_dist_val)

    # Best placement from next agent position
    next_best_place_y = jnp.float32(0)
    next_best_place_x = jnp.float32(0)
    next_best_dist_val = jnp.float32(1000.0)
    for d in range(4):
        hny = jnp.clip(hex_y_int + DIRECTIONS[d][0], 0, H - 1)
        hnx = jnp.clip(hex_x_int + DIRECTIONS[d][1], 0, W - 1)
        is_floor = (grid[hny, hnx, 0] == FLOOR)
        dist_d = jnp.abs(next_agent_y - jnp.float32(hny)) + jnp.abs(next_agent_x - jnp.float32(hnx))
        use_this = is_floor & (dist_d < next_best_dist_val)
        next_best_place_y = jax.lax.select(use_this, jnp.float32(hny), next_best_place_y)
        next_best_place_x = jax.lax.select(use_this, jnp.float32(hnx), next_best_place_x)
        next_best_dist_val = jax.lax.select(use_this, dist_d, next_best_dist_val)

    # Milestone 1: pickup blue pyramid -> green ball
    just_picked_up = (~was_holding_green_ball) & now_holding_green_ball
    pickup_reward = jax.lax.select(just_picked_up, jnp.float32(5.0), jnp.float32(0.0))

    # Green ball on ground in next state
    gb_mask_next = (next_grid[:, :, 0] == BALL) & (next_grid[:, :, 1] == GREEN)
    gb_count_next = jnp.sum(gb_mask_next).astype(jnp.float32)
    gb_y_next = jnp.sum(yy * gb_mask_next).astype(jnp.float32) / jnp.maximum(gb_count_next, 1.0)
    gb_x_next = jnp.sum(xx * gb_mask_next).astype(jnp.float32) / jnp.maximum(gb_count_next, 1.0)

    dist_gb_hex_next = jnp.abs(gb_y_next - hex_y) + jnp.abs(gb_x_next - hex_x)
    gb_adj_to_hex_next = (dist_gb_hex_next <= 1.0) & (gb_count_next > 0.0)

    # Success: placed ball adjacent to hex
    ball_just_placed = was_holding_green_ball & (~now_holding_green_ball)
    success = ball_just_placed & gb_adj_to_hex_next
    success_reward = jax.lax.select(success, jnp.float32(20.0), jnp.float32(0.0))

    # Wrong placement penalty
    wrong_place = ball_just_placed & (~gb_adj_to_hex_next)
    wrong_penalty = jax.lax.select(wrong_place, jnp.float32(-5.0), jnp.float32(0.0))

    # Phase 1 shaping: move toward pyramid
    dist_to_pyr_cur = jnp.abs(agent_y - pyr_y) + jnp.abs(agent_x - pyr_x)
    dist_to_pyr_next = jnp.abs(next_agent_y - pyr_y) + jnp.abs(next_agent_x - pyr_x)
    pyr_progress = (dist_to_pyr_cur - dist_to_pyr_next) / norm_denom
    pyr_shaping = jax.lax.select(
        (~was_holding_green_ball) & (blue_pyr_count > 0.0),
        pyr_progress * jnp.float32(3.0),
        jnp.float32(0.0)
    )

    # Phase 2 shaping: move toward hex (direct hex shaping)
    dist_to_hex_cur = jnp.abs(agent_y - hex_y) + jnp.abs(agent_x - hex_x)
    dist_to_hex_next = jnp.abs(next_agent_y - hex_y) + jnp.abs(next_agent_x - hex_x)
    hex_progress = (dist_to_hex_cur - dist_to_hex_next) / norm_denom
    hex_shaping = jax.lax.select(
        was_holding_green_ball & (yellow_hex_count > 0.0),
        hex_progress * jnp.float32(3.0),
        jnp.float32(0.0)
    )

    # Phase 2 additional shaping: move toward best placement spot
    dist_to_place_cur = jnp.abs(agent_y - best_place_y) + jnp.abs(agent_x - best_place_x)
    dist_to_place_next = jnp.abs(next_agent_y - next_best_place_y) + jnp.abs(next_agent_x - next_best_place_x)
    place_progress = (dist_to_place_cur - dist_to_place_next) / norm_denom
    place_shaping = jax.lax.select(
        was_holding_green_ball & (yellow_hex_count > 0.0),
        place_progress * jnp.float32(1.5),
        jnp.float32(0.0)
    )

    # Bonus: first time reaching adjacency to hex while holding
    next_adj_to_hex = (jnp.abs(next_agent_y - hex_y) + jnp.abs(next_agent_x - hex_x)) <= 1.0
    prev_adj_to_hex = (jnp.abs(agent_y - hex_y) + jnp.abs(agent_x - hex_x)) <= 1.0
    just_reached_adj = was_holding_green_ball & next_adj_to_hex & (~prev_adj_to_hex)
    adj_bonus = jax.lax.select(just_reached_adj, jnp.float32(3.0), jnp.float32(0.0))

    # Bonus: agent is positioned to putdown (holding, facing floor adj to hex)
    next_dy = DIRECTIONS[next_state.agent.direction][0]
    next_dx = DIRECTIONS[next_state.agent.direction][1]
    next_front_y = jnp.clip(next_state.agent.position[0].astype(jnp.int32) + next_dy, 0, H - 1)
    next_front_x = jnp.clip(next_state.agent.position[1].astype(jnp.int32) + next_dx, 0, W - 1)
    next_front_is_floor = (next_grid[next_front_y, next_front_x, 0] == FLOOR)
    next_front_adj_hex = (jnp.abs(jnp.float32(next_front_y) - hex_y) + jnp.abs(jnp.float32(next_front_x) - hex_x)) <= 1.0
    ready_to_putdown = now_holding_green_ball & next_front_is_floor & next_front_adj_hex
    ready_bonus = jax.lax.select(ready_to_putdown, jnp.float32(0.5), jnp.float32(0.0))

    # Small step penalty for efficiency
    step_penalty = jnp.float32(-0.01)

    total_reward = (
        pickup_reward
        + success_reward
        + wrong_penalty
        + pyr_shaping
        + hex_shaping
        + place_shaping
        + adj_bonus
        + ready_bonus
        + step_penalty
    )

    return total_reward.astype(jnp.float32)
