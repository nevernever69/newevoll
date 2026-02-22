"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Pick up the blue pyramid. Grid 9x9, max_steps=80.
Success rate: 99%
Obs dim:      174
Generation:   1
Iteration:    30
Mode:         full
"""

import jax
import jax.numpy as jnp

# Tile and color constants
EMPTY = 0
FLOOR = 1
WALL = 2
BALL = 3
SQUARE = 4
PYRAMID = 5
GOAL = 6
KEY = 7
DOOR_LOCKED = 8
DOOR_CLOSED = 9
DOOR_OPEN = 10
HEX = 11
STAR = 12

BLUE = 3

DIRECTIONS = jnp.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=jnp.int32)


def get_observation(state):
    H, W = state.grid.shape[:2]
    H_f = jnp.float32(H)
    W_f = jnp.float32(W)

    # Agent position (normalized)
    agent_y_raw = state.agent.position[0].astype(jnp.float32)
    agent_x_raw = state.agent.position[1].astype(jnp.float32)
    agent_y = agent_y_raw / (H_f - 1.0)
    agent_x = agent_x_raw / (W_f - 1.0)

    # Agent direction one-hot
    dir_oh = jax.nn.one_hot(state.agent.direction, 4)  # shape (4,)

    # Pocket info
    pocket_tile = state.agent.pocket[0].astype(jnp.float32) / 12.0
    pocket_color = state.agent.pocket[1].astype(jnp.float32) / 11.0
    pocket_empty = (state.agent.pocket[0] == 0).astype(jnp.float32)
    holding_blue_pyramid = (
        (state.agent.pocket[0] == PYRAMID) & (state.agent.pocket[1] == BLUE)
    ).astype(jnp.float32)

    # Find the blue pyramid location
    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')  # (H, W)

    blue_pyramid_mask = (
        (state.grid[:, :, 0] == PYRAMID) & (state.grid[:, :, 1] == BLUE)
    )
    count_bp = jnp.sum(blue_pyramid_mask).astype(jnp.float32)

    bp_y = jnp.sum(yy * blue_pyramid_mask).astype(jnp.float32) / jnp.maximum(count_bp, 1.0)
    bp_x = jnp.sum(xx * blue_pyramid_mask).astype(jnp.float32) / jnp.maximum(count_bp, 1.0)

    bp_y_norm = bp_y / (H_f - 1.0)
    bp_x_norm = bp_x / (W_f - 1.0)

    # Relative position from agent to blue pyramid (normalized)
    rel_y = (bp_y - agent_y_raw) / (H_f - 1.0)
    rel_x = (bp_x - agent_x_raw) / (W_f - 1.0)

    # Manhattan distance to blue pyramid (normalized)
    manhattan_to_bp = jnp.abs(agent_y_raw - bp_y) + jnp.abs(agent_x_raw - bp_x)
    dist_bp = manhattan_to_bp / (H_f + W_f - 2.0)

    # Whether blue pyramid exists on grid
    bp_exists = (count_bp > 0).astype(jnp.float32)

    # Is agent adjacent to blue pyramid?
    adjacent_to_bp = (manhattan_to_bp <= 1.0).astype(jnp.float32) * bp_exists

    # Sign of relative direction (useful for knowing which way to turn)
    sign_dy = jnp.sign(bp_y - agent_y_raw)  # -1, 0, or 1
    sign_dx = jnp.sign(bp_x - agent_x_raw)  # -1, 0, or 1

    # Directional indicators
    bp_is_up = (sign_dy < 0).astype(jnp.float32)
    bp_is_down = (sign_dy > 0).astype(jnp.float32)
    bp_is_left = (sign_dx < 0).astype(jnp.float32)
    bp_is_right = (sign_dx > 0).astype(jnp.float32)

    # Front tile info
    dy = DIRECTIONS[state.agent.direction, 0]
    dx = DIRECTIONS[state.agent.direction, 1]
    front_y = jnp.clip(state.agent.position[0].astype(jnp.int32) + dy, 0, H - 1)
    front_x = jnp.clip(state.agent.position[1].astype(jnp.int32) + dx, 0, W - 1)
    front_tile_id = state.grid[front_y, front_x, 0].astype(jnp.float32) / 12.0
    front_tile_color = state.grid[front_y, front_x, 1].astype(jnp.float32) / 11.0
    front_is_blue_pyramid = (
        (state.grid[front_y, front_x, 0] == PYRAMID) &
        (state.grid[front_y, front_x, 1] == BLUE)
    ).astype(jnp.float32)

    # Step progress
    step_norm = state.step_num.astype(jnp.float32) / 80.0

    # Local 7x7 view around agent
    view_size = 7
    half = view_size // 2
    agent_row = state.agent.position[0].astype(jnp.int32)
    agent_col = state.agent.position[1].astype(jnp.int32)

    offsets = jnp.arange(-half, half + 1)
    oy, ox = jnp.meshgrid(offsets, offsets, indexing='ij')  # (7, 7)

    rows = jnp.clip(agent_row + oy, 0, H - 1)
    cols = jnp.clip(agent_col + ox, 0, W - 1)

    local_tiles = state.grid[rows, cols, 0].astype(jnp.float32) / 12.0  # (7, 7)
    local_colors = state.grid[rows, cols, 1].astype(jnp.float32) / 11.0  # (7, 7)

    # Blue pyramid indicator in local view
    local_is_bp = (
        (state.grid[rows, cols, 0] == PYRAMID) &
        (state.grid[rows, cols, 1] == BLUE)
    ).astype(jnp.float32)

    local_tiles_flat = local_tiles.reshape(-1)    # 49
    local_colors_flat = local_colors.reshape(-1)  # 49
    local_bp_flat = local_is_bp.reshape(-1)       # 49

    obs = jnp.concatenate([
        jnp.array([agent_y, agent_x], dtype=jnp.float32),                                              # 2
        dir_oh.astype(jnp.float32),                                                                     # 4
        jnp.array([pocket_tile, pocket_color, pocket_empty, holding_blue_pyramid], dtype=jnp.float32), # 4
        jnp.array([bp_y_norm, bp_x_norm, rel_y, rel_x, dist_bp, bp_exists], dtype=jnp.float32),       # 6
        jnp.array([adjacent_to_bp, front_is_blue_pyramid, sign_dy, sign_dx], dtype=jnp.float32),      # 4
        jnp.array([bp_is_up, bp_is_down, bp_is_left, bp_is_right], dtype=jnp.float32),                # 4
        jnp.array([front_tile_id, front_tile_color], dtype=jnp.float32),                               # 2
        jnp.array([step_norm], dtype=jnp.float32),                                                     # 1
        local_tiles_flat,                                                                               # 49
        local_colors_flat,                                                                              # 49
        local_bp_flat,                                                                                  # 49
    ])
    # Total: 2+4+4+6+4+4+2+1+49+49+49 = 174
    return obs.astype(jnp.float32)


def compute_reward(state, action, next_state):
    H, W = state.grid.shape[:2]
    H_f = jnp.float32(H)
    W_f = jnp.float32(W)

    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    # Check holding states
    holding_now = (
        (next_state.agent.pocket[0] == PYRAMID) &
        (next_state.agent.pocket[1] == BLUE)
    ).astype(jnp.float32)

    was_holding = (
        (state.agent.pocket[0] == PYRAMID) &
        (state.agent.pocket[1] == BLUE)
    ).astype(jnp.float32)

    # Big reward for picking up the blue pyramid (task completion)
    pickup_reward = jax.lax.select(
        (holding_now > 0.5) & (was_holding < 0.5),
        jnp.float32(10.0),
        jnp.float32(0.0)
    )

    # Blue pyramid location in current state
    bp_mask_cur = (state.grid[:, :, 0] == PYRAMID) & (state.grid[:, :, 1] == BLUE)
    count_cur = jnp.sum(bp_mask_cur).astype(jnp.float32)
    bp_y_cur = jnp.sum(yy * bp_mask_cur).astype(jnp.float32) / jnp.maximum(count_cur, 1.0)
    bp_x_cur = jnp.sum(xx * bp_mask_cur).astype(jnp.float32) / jnp.maximum(count_cur, 1.0)

    # Blue pyramid location in next state
    bp_mask_nxt = (next_state.grid[:, :, 0] == PYRAMID) & (next_state.grid[:, :, 1] == BLUE)
    count_nxt = jnp.sum(bp_mask_nxt).astype(jnp.float32)
    bp_y_nxt = jnp.sum(yy * bp_mask_nxt).astype(jnp.float32) / jnp.maximum(count_nxt, 1.0)
    bp_x_nxt = jnp.sum(xx * bp_mask_nxt).astype(jnp.float32) / jnp.maximum(count_nxt, 1.0)

    # Distance before and after
    dist_before = (
        jnp.abs(state.agent.position[0].astype(jnp.float32) - bp_y_cur) +
        jnp.abs(state.agent.position[1].astype(jnp.float32) - bp_x_cur)
    )
    dist_after = (
        jnp.abs(next_state.agent.position[0].astype(jnp.float32) - bp_y_nxt) +
        jnp.abs(next_state.agent.position[1].astype(jnp.float32) - bp_x_nxt)
    )

    # Dense approach shaping (only when not holding)
    not_holding = (1.0 - was_holding)
    shaping_reward = not_holding * (dist_before - dist_after) * 0.5

    # Bonus for transitioning to adjacent state
    manhattan_next = dist_after
    manhattan_cur = dist_before

    became_adjacent = jax.lax.select(
        (manhattan_next <= 1.0) & (manhattan_cur > 1.0) & (not_holding > 0.5),
        jnp.float32(0.5),
        jnp.float32(0.0)
    )

    # Reward for facing the blue pyramid (ready to pick up)
    dy_next = DIRECTIONS[next_state.agent.direction, 0]
    dx_next = DIRECTIONS[next_state.agent.direction, 1]
    front_y_next = jnp.clip(next_state.agent.position[0].astype(jnp.int32) + dy_next, 0, H - 1)
    front_x_next = jnp.clip(next_state.agent.position[1].astype(jnp.int32) + dx_next, 0, W - 1)
    front_is_bp_next = (
        (next_state.grid[front_y_next, front_x_next, 0] == PYRAMID) &
        (next_state.grid[front_y_next, front_x_next, 1] == BLUE)
    ).astype(jnp.float32)

    dy_cur = DIRECTIONS[state.agent.direction, 0]
    dx_cur = DIRECTIONS[state.agent.direction, 1]
    front_y_cur = jnp.clip(state.agent.position[0].astype(jnp.int32) + dy_cur, 0, H - 1)
    front_x_cur = jnp.clip(state.agent.position[1].astype(jnp.int32) + dx_cur, 0, W - 1)
    front_is_bp_cur = (
        (state.grid[front_y_cur, front_x_cur, 0] == PYRAMID) &
        (state.grid[front_y_cur, front_x_cur, 1] == BLUE)
    ).astype(jnp.float32)

    # Reward for transitioning to facing the pyramid
    facing_reward = jax.lax.select(
        (front_is_bp_next > 0.5) & (front_is_bp_cur < 0.5) & (not_holding > 0.5),
        jnp.float32(1.0),
        jnp.float32(0.0)
    )

    # Small step penalty for efficiency
    step_penalty = jnp.float32(-0.005)

    total_reward = pickup_reward + shaping_reward + became_adjacent + facing_reward + step_penalty
    return total_reward.astype(jnp.float32)
