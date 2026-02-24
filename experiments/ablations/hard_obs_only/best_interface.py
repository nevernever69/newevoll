"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Pick up blue pyramid (transforms to green ball in hand), then place green ball next to yellow hex. 4-room 13x13 grid with distractors, max_steps=400.
Success rate: 34%
Obs dim:      472
Generation:   1
Iteration:    30
Mode:         obs_only
"""

import jax
import jax.numpy as jnp

def get_observation(state) -> jnp.ndarray:
    H, W = state.grid.shape[:2]
    H_f = jnp.float32(H)
    W_f = jnp.float32(W)

    DIRECTIONS = jnp.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=jnp.int32)

    agent_y = state.agent.position[0].astype(jnp.float32)
    agent_x = state.agent.position[1].astype(jnp.float32)
    norm_ay = agent_y / (H_f - 1)
    norm_ax = agent_x / (W_f - 1)
    dir_oh = jax.nn.one_hot(state.agent.direction, 4)

    tile_ids = state.grid[:, :, 0]
    tile_colors = state.grid[:, :, 1]

    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    yy_f = yy.astype(jnp.float32)
    xx_f = xx.astype(jnp.float32)

    # ---- Pocket ----
    pocket_empty = (state.agent.pocket[0] == 0).astype(jnp.float32)
    holding_green_ball = ((state.agent.pocket[0] == 3) & (state.agent.pocket[1] == 2)).astype(jnp.float32)
    pocket_tile_oh = jax.nn.one_hot(state.agent.pocket[0], 13)
    pocket_color_oh = jax.nn.one_hot(state.agent.pocket[1], 12)

    # ---- Blue pyramid (tile=5, color=3) ----
    blue_pyr_mask = (tile_ids == 5) & (tile_colors == 3)
    blue_pyr_count = jnp.sum(blue_pyr_mask).astype(jnp.float32)
    blue_pyr_y = jnp.sum(yy_f * blue_pyr_mask) / jnp.maximum(blue_pyr_count, 1.0)
    blue_pyr_x = jnp.sum(xx_f * blue_pyr_mask) / jnp.maximum(blue_pyr_count, 1.0)
    blue_pyr_exists = jnp.clip(blue_pyr_count, 0.0, 1.0)
    rel_pyr_y = (blue_pyr_y - agent_y) / (H_f - 1)
    rel_pyr_x = (blue_pyr_x - agent_x) / (W_f - 1)
    dist_pyr = (jnp.abs(blue_pyr_y - agent_y) + jnp.abs(blue_pyr_x - agent_x)) / (H_f + W_f - 2)

    # ---- Yellow hex (tile=11, color=5) ----
    yel_hex_mask = (tile_ids == 11) & (tile_colors == 5)
    yel_hex_count = jnp.sum(yel_hex_mask).astype(jnp.float32)
    yel_hex_y = jnp.sum(yy_f * yel_hex_mask) / jnp.maximum(yel_hex_count, 1.0)
    yel_hex_x = jnp.sum(xx_f * yel_hex_mask) / jnp.maximum(yel_hex_count, 1.0)
    yel_hex_exists = jnp.clip(yel_hex_count, 0.0, 1.0)
    rel_hex_y = (yel_hex_y - agent_y) / (H_f - 1)
    rel_hex_x = (yel_hex_x - agent_x) / (W_f - 1)
    dist_hex = (jnp.abs(yel_hex_y - agent_y) + jnp.abs(yel_hex_x - agent_x)) / (H_f + W_f - 2)

    # ---- Task phases ----
    phase_pickup = (blue_pyr_exists * pocket_empty)
    phase_place = holding_green_ball

    # ---- Floor tiles adjacent to yellow hex (placement targets) ----
    adj_to_hex = ((jnp.abs(yy_f - yel_hex_y) + jnp.abs(xx_f - yel_hex_x)) == 1.0)
    floor_adj_hex_mask = adj_to_hex & (tile_ids == 1)
    floor_adj_count = jnp.sum(floor_adj_hex_mask).astype(jnp.float32)
    place_y = jnp.sum(yy_f * floor_adj_hex_mask) / jnp.maximum(floor_adj_count, 1.0)
    place_x = jnp.sum(xx_f * floor_adj_hex_mask) / jnp.maximum(floor_adj_count, 1.0)
    rel_place_y = (place_y - agent_y) / (H_f - 1)
    rel_place_x = (place_x - agent_x) / (W_f - 1)
    dist_place = (jnp.abs(place_y - agent_y) + jnp.abs(place_x - agent_x)) / (H_f + W_f - 2)

    # Nearest placement spot to agent
    place_dist_map = jnp.where(floor_adj_hex_mask,
                                jnp.abs(yy_f - agent_y) + jnp.abs(xx_f - agent_x),
                                9999.0)
    nearest_place_flat = jnp.argmin(place_dist_map)
    nearest_place_y = (nearest_place_flat // W).astype(jnp.float32)
    nearest_place_x = (nearest_place_flat % W).astype(jnp.float32)
    rel_nearest_place_y = (nearest_place_y - agent_y) / (H_f - 1)
    rel_nearest_place_x = (nearest_place_x - agent_x) / (W_f - 1)
    dist_nearest_place = (jnp.abs(nearest_place_y - agent_y) + jnp.abs(nearest_place_x - agent_x)) / (H_f + W_f - 2)

    # ---- Phase-conditioned target ----
    target_y = jax.lax.select(phase_place > 0.5, nearest_place_y, blue_pyr_y)
    target_x = jax.lax.select(phase_place > 0.5, nearest_place_x, blue_pyr_x)
    rel_target_y = (target_y - agent_y) / (H_f - 1)
    rel_target_x = (target_x - agent_x) / (W_f - 1)
    dist_target = (jnp.abs(target_y - agent_y) + jnp.abs(target_x - agent_x)) / (H_f + W_f - 2)

    # Direction to target
    target_dy = target_y - agent_y
    target_dx = target_x - agent_x
    go_up = (target_dy < 0).astype(jnp.float32)
    go_down = (target_dy > 0).astype(jnp.float32)
    go_left = (target_dx < 0).astype(jnp.float32)
    go_right = (target_dx > 0).astype(jnp.float32)
    facing_target = jnp.sum(jnp.array([
        go_up * (state.agent.direction == 0).astype(jnp.float32),
        go_right * (state.agent.direction == 1).astype(jnp.float32),
        go_down * (state.agent.direction == 2).astype(jnp.float32),
        go_left * (state.agent.direction == 3).astype(jnp.float32),
    ]))

    # ---- Front tile ----
    dy_f = DIRECTIONS[state.agent.direction][0]
    dx_f = DIRECTIONS[state.agent.direction][1]
    front_y = jnp.clip(state.agent.position[0].astype(jnp.int32) + dy_f, 0, H - 1)
    front_x = jnp.clip(state.agent.position[1].astype(jnp.int32) + dx_f, 0, W - 1)
    front_tile_id = tile_ids[front_y, front_x]
    front_color_id = tile_colors[front_y, front_x]
    front_tile_oh = jax.nn.one_hot(front_tile_id, 13)
    front_color_oh = jax.nn.one_hot(front_color_id, 12)
    front_is_blue_pyr = ((front_tile_id == 5) & (front_color_id == 3)).astype(jnp.float32)
    front_is_floor = (front_tile_id == 1).astype(jnp.float32)
    front_is_closed_door = (front_tile_id == 9).astype(jnp.float32)
    front_is_locked_door = (front_tile_id == 8).astype(jnp.float32)
    front_is_open_door = (front_tile_id == 10).astype(jnp.float32)
    front_to_hex_dist = (jnp.abs(front_y.astype(jnp.float32) - yel_hex_y) +
                         jnp.abs(front_x.astype(jnp.float32) - yel_hex_x))
    front_adj_hex = (front_to_hex_dist <= 1.0).astype(jnp.float32)
    front_floor_adj_hex = front_is_floor * front_adj_hex
    can_place_now = holding_green_ball * front_floor_adj_hex
    can_pickup_now = front_is_blue_pyr * pocket_empty

    # Is front tile the nearest placement target?
    front_is_nearest_place = ((front_y.astype(jnp.float32) == nearest_place_y) &
                               (front_x.astype(jnp.float32) == nearest_place_x)).astype(jnp.float32)

    # ---- Adjacency ----
    agent_adj_hex = ((jnp.abs(agent_y - yel_hex_y) + jnp.abs(agent_x - yel_hex_x)) <= 1.0).astype(jnp.float32)
    agent_adj_pyr = ((jnp.abs(agent_y - blue_pyr_y) + jnp.abs(agent_x - blue_pyr_x)) <= 1.0).astype(jnp.float32)
    agent_at_place = ((jnp.abs(agent_y - nearest_place_y) + jnp.abs(agent_x - nearest_place_x)) <= 1.0).astype(jnp.float32)

    # ---- All 4 neighbors of hex for placement ----
    hex_y_int = yel_hex_y.astype(jnp.int32)
    hex_x_int = yel_hex_x.astype(jnp.int32)
    n_up_y = jnp.clip(hex_y_int - 1, 0, H - 1)
    n_dn_y = jnp.clip(hex_y_int + 1, 0, H - 1)
    n_lt_x = jnp.clip(hex_x_int - 1, 0, W - 1)
    n_rt_x = jnp.clip(hex_x_int + 1, 0, W - 1)
    hex_up_floor = (tile_ids[n_up_y, hex_x_int] == 1).astype(jnp.float32)
    hex_dn_floor = (tile_ids[n_dn_y, hex_x_int] == 1).astype(jnp.float32)
    hex_lt_floor = (tile_ids[hex_y_int, n_lt_x] == 1).astype(jnp.float32)
    hex_rt_floor = (tile_ids[hex_y_int, n_rt_x] == 1).astype(jnp.float32)
    rel_hex_up_y = (n_up_y.astype(jnp.float32) - agent_y) / (H_f - 1)
    rel_hex_up_x = (hex_x_int.astype(jnp.float32) - agent_x) / (W_f - 1)
    rel_hex_dn_y = (n_dn_y.astype(jnp.float32) - agent_y) / (H_f - 1)
    rel_hex_dn_x = (hex_x_int.astype(jnp.float32) - agent_x) / (W_f - 1)
    rel_hex_lt_y = (hex_y_int.astype(jnp.float32) - agent_y) / (H_f - 1)
    rel_hex_lt_x = (n_lt_x.astype(jnp.float32) - agent_x) / (W_f - 1)
    rel_hex_rt_y = (hex_y_int.astype(jnp.float32) - agent_y) / (H_f - 1)
    rel_hex_rt_x = (n_rt_x.astype(jnp.float32) - agent_x) / (W_f - 1)

    # ---- Door analysis ----
    closed_door_mask = (tile_ids == 9)
    locked_door_mask = (tile_ids == 8)
    open_door_mask = (tile_ids == 10)
    all_blockable = closed_door_mask | locked_door_mask
    n_closed = jnp.sum(closed_door_mask).astype(jnp.float32)
    n_locked = jnp.sum(locked_door_mask).astype(jnp.float32)
    n_open = jnp.sum(open_door_mask).astype(jnp.float32)

    # Nearest blockable door to agent
    dist_to_all = jnp.abs(yy_f - agent_y) + jnp.abs(xx_f - agent_x)
    door_dist_map = jnp.where(all_blockable, dist_to_all, 999.0)
    nearest_door_dist = jnp.min(door_dist_map)
    nearest_door_flat = jnp.argmin(door_dist_map)
    nearest_door_y = (nearest_door_flat // W).astype(jnp.float32)
    nearest_door_x = (nearest_door_flat % W).astype(jnp.float32)
    rel_nearest_door_y = (nearest_door_y - agent_y) / (H_f - 1)
    rel_nearest_door_x = (nearest_door_x - agent_x) / (W_f - 1)
    nearest_door_dist_norm = nearest_door_dist / (H_f + W_f - 2)
    has_blockable_door = (jnp.sum(all_blockable) > 0).astype(jnp.float32)

    # Nearest door toward current target
    dist_to_target_map = jnp.abs(yy_f - target_y) + jnp.abs(xx_f - target_x)
    door_toward_target = jnp.where(all_blockable, dist_to_target_map, 999.0)
    nearest_target_door_flat = jnp.argmin(door_toward_target)
    nearest_target_door_y = (nearest_target_door_flat // W).astype(jnp.float32)
    nearest_target_door_x = (nearest_target_door_flat % W).astype(jnp.float32)
    rel_target_door_y = (nearest_target_door_y - agent_y) / (H_f - 1)
    rel_target_door_x = (nearest_target_door_x - agent_x) / (W_f - 1)

    # ---- Room quadrant ----
    mid_y = H_f / 2.0
    mid_x = W_f / 2.0
    agent_room = jnp.array([
        ((agent_y < mid_y) & (agent_x < mid_x)).astype(jnp.float32),
        ((agent_y < mid_y) & (agent_x >= mid_x)).astype(jnp.float32),
        ((agent_y >= mid_y) & (agent_x < mid_x)).astype(jnp.float32),
        ((agent_y >= mid_y) & (agent_x >= mid_x)).astype(jnp.float32),
    ])
    pyr_room = jnp.array([
        ((blue_pyr_y < mid_y) & (blue_pyr_x < mid_x)).astype(jnp.float32),
        ((blue_pyr_y < mid_y) & (blue_pyr_x >= mid_x)).astype(jnp.float32),
        ((blue_pyr_y >= mid_y) & (blue_pyr_x < mid_x)).astype(jnp.float32),
        ((blue_pyr_y >= mid_y) & (blue_pyr_x >= mid_x)).astype(jnp.float32),
    ])
    hex_room = jnp.array([
        ((yel_hex_y < mid_y) & (yel_hex_x < mid_x)).astype(jnp.float32),
        ((yel_hex_y < mid_y) & (yel_hex_x >= mid_x)).astype(jnp.float32),
        ((yel_hex_y >= mid_y) & (yel_hex_x < mid_x)).astype(jnp.float32),
        ((yel_hex_y >= mid_y) & (yel_hex_x >= mid_x)).astype(jnp.float32),
    ])
    same_room_pyr = jnp.sum(agent_room * pyr_room)
    same_room_hex = jnp.sum(agent_room * hex_room)
    same_room_target = jax.lax.select(phase_place > 0.5, same_room_hex, same_room_pyr)

    # ---- Step ----
    step_norm = state.step_num.astype(jnp.float32) / 400.0

    # ---- Full grid semantic maps ----
    # Blue pyramid locations (169)
    blue_pyr_map = blue_pyr_mask.astype(jnp.float32).flatten()
    # Yellow hex locations (169)
    yel_hex_map = yel_hex_mask.astype(jnp.float32).flatten()

    # ---- Structured features ----
    structured = jnp.concatenate([
        # Agent state (6)
        jnp.array([norm_ay, norm_ax]),
        dir_oh,

        # Pocket (13+12+2 = 27)
        pocket_tile_oh,
        pocket_color_oh,
        jnp.array([pocket_empty, holding_green_ball]),

        # Task phase + action readiness (5)
        jnp.array([phase_pickup, phase_place, can_pickup_now, can_place_now,
                   holding_green_ball * front_is_nearest_place]),

        # Phase-conditioned target (3)
        jnp.array([rel_target_y, rel_target_x, dist_target]),

        # Direction to target (5)
        jnp.array([go_up, go_down, go_left, go_right, facing_target]),

        # Blue pyramid absolute + relative (6)
        jnp.array([blue_pyr_y / (H_f - 1), blue_pyr_x / (W_f - 1),
                   rel_pyr_y, rel_pyr_x, dist_pyr, blue_pyr_exists]),

        # Yellow hex absolute + relative (6)
        jnp.array([yel_hex_y / (H_f - 1), yel_hex_x / (W_f - 1),
                   rel_hex_y, rel_hex_x, dist_hex, yel_hex_exists]),

        # Placement target - mean + nearest (6)
        jnp.array([rel_place_y, rel_place_x, dist_place,
                   rel_nearest_place_y, rel_nearest_place_x, dist_nearest_place]),

        # Hex neighbor floors + relative positions (12)
        jnp.array([hex_up_floor, rel_hex_up_y, rel_hex_up_x,
                   hex_dn_floor, rel_hex_dn_y, rel_hex_dn_x,
                   hex_lt_floor, rel_hex_lt_y, rel_hex_lt_x,
                   hex_rt_floor, rel_hex_rt_y, rel_hex_rt_x]),

        # Adjacency flags (3)
        jnp.array([agent_adj_hex, agent_adj_pyr, agent_at_place]),

        # Front tile (13+12+6 = 31)
        front_tile_oh,
        front_color_oh,
        jnp.array([front_is_blue_pyr, front_floor_adj_hex, front_is_floor,
                   front_is_closed_door, front_is_locked_door, front_is_open_door]),

        # Room info (4+4+4+3 = 15)
        agent_room,
        pyr_room,
        hex_room,
        jnp.array([same_room_pyr, same_room_hex, same_room_target]),

        # Door navigation (8)
        jnp.array([rel_nearest_door_y, rel_nearest_door_x,
                   nearest_door_dist_norm, has_blockable_door,
                   rel_target_door_y, rel_target_door_x,
                   n_closed / 4.0, n_open / 4.0]),

        # Step (1)
        jnp.array([step_norm]),
    ])
    # 6+27+5+3+5+6+6+6+12+3+31+15+8+1 = 134

    # Full grid maps: blue_pyr_map (169) + yel_hex_map (169) = 338
    # Total: 134 + 338 = 472 (under 512)
    obs = jnp.concatenate([structured, blue_pyr_map, yel_hex_map])

    return obs.astype(jnp.float32)
