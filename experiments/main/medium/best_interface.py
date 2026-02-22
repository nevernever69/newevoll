"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Place the yellow pyramid adjacent to the green square. Grid 9x9, max_steps=80.
Success rate: 97%
Obs dim:      102
Generation:   3
Iteration:    30
Mode:         full
"""

import jax
import jax.numpy as jnp

FLOOR = 1
WALL = 2
SQUARE = 4
PYRAMID = 5

YELLOW = 5
GREEN = 2

DIRECTIONS = jnp.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=jnp.int32)


def get_observation(state):
    H, W = state.grid.shape[:2]
    H_f = jnp.float32(H)
    W_f = jnp.float32(W)

    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    # Agent
    agent_y = state.agent.position[0].astype(jnp.float32)
    agent_x = state.agent.position[1].astype(jnp.float32)
    norm_agent_y = agent_y / (H_f - 1)
    norm_agent_x = agent_x / (W_f - 1)
    dir_oh = jax.nn.one_hot(state.agent.direction, 4)

    # Yellow pyramid on grid
    yp_mask = (state.grid[:, :, 0] == PYRAMID) & (state.grid[:, :, 1] == YELLOW)
    yp_count = jnp.sum(yp_mask).astype(jnp.float32)
    yp_y = jnp.sum(yy * yp_mask).astype(jnp.float32) / jnp.maximum(yp_count, 1.0)
    yp_x = jnp.sum(xx * yp_mask).astype(jnp.float32) / jnp.maximum(yp_count, 1.0)
    yp_on_grid = (yp_count > 0).astype(jnp.float32)

    # Green square on grid
    gs_mask = (state.grid[:, :, 0] == SQUARE) & (state.grid[:, :, 1] == GREEN)
    gs_count = jnp.sum(gs_mask).astype(jnp.float32)
    gs_y = jnp.sum(yy * gs_mask).astype(jnp.float32) / jnp.maximum(gs_count, 1.0)
    gs_x = jnp.sum(xx * gs_mask).astype(jnp.float32) / jnp.maximum(gs_count, 1.0)

    # Holding yellow pyramid
    holding_yp = ((state.agent.pocket[0] == PYRAMID) & (state.agent.pocket[1] == YELLOW)).astype(jnp.float32)
    pocket_empty = (state.agent.pocket[0] == 0).astype(jnp.float32)

    # Normalized object positions
    norm_yp_y = yp_y / (H_f - 1)
    norm_yp_x = yp_x / (W_f - 1)
    norm_gs_y = gs_y / (H_f - 1)
    norm_gs_x = gs_x / (W_f - 1)

    # Relative: agent -> yellow pyramid
    rel_agent_yp_y = (yp_y - agent_y) / H_f
    rel_agent_yp_x = (yp_x - agent_x) / W_f

    # Relative: agent -> green square
    rel_agent_gs_y = (gs_y - agent_y) / H_f
    rel_agent_gs_x = (gs_x - agent_x) / W_f

    # Relative: yellow pyramid -> green square
    rel_yp_gs_y = (gs_y - yp_y) / H_f
    rel_yp_gs_x = (gs_x - yp_x) / W_f

    # Manhattan distances (normalized)
    dist_agent_yp = (jnp.abs(agent_y - yp_y) + jnp.abs(agent_x - yp_x)) / (H_f + W_f)
    dist_agent_gs = (jnp.abs(agent_y - gs_y) + jnp.abs(agent_x - gs_x)) / (H_f + W_f)
    dist_yp_gs = (jnp.abs(yp_y - gs_y) + jnp.abs(yp_x - gs_x)) / (H_f + W_f)

    # Front tile
    dy = DIRECTIONS[state.agent.direction, 0]
    dx = DIRECTIONS[state.agent.direction, 1]
    front_y = jnp.clip(state.agent.position[0].astype(jnp.int32) + dy, 0, H - 1)
    front_x = jnp.clip(state.agent.position[1].astype(jnp.int32) + dx, 0, W - 1)
    front_tile_id = state.grid[front_y, front_x, 0].astype(jnp.float32) / 12.0
    front_tile_color = state.grid[front_y, front_x, 1].astype(jnp.float32) / 11.0
    front_is_yp = ((state.grid[front_y, front_x, 0] == PYRAMID) & (state.grid[front_y, front_x, 1] == YELLOW)).astype(jnp.float32)
    front_is_gs = ((state.grid[front_y, front_x, 0] == SQUARE) & (state.grid[front_y, front_x, 1] == GREEN)).astype(jnp.float32)
    front_is_floor = (state.grid[front_y, front_x, 0] == FLOOR).astype(jnp.float32)

    # Front tile distance to green square (for putdown guidance)
    front_dist_to_gs = jnp.abs(front_y.astype(jnp.float32) - gs_y) + jnp.abs(front_x.astype(jnp.float32) - gs_x)
    front_adj_to_gs = (front_dist_to_gs <= 1.0).astype(jnp.float32)
    # Is front tile a floor cell adjacent to green square? (ideal putdown spot)
    front_ideal_putdown = (front_is_floor * front_adj_to_gs)

    # Is agent adjacent to green square?
    dist_agent_gs_raw = jnp.abs(agent_y - gs_y) + jnp.abs(agent_x - gs_x)
    agent_adj_gs = (dist_agent_gs_raw <= 1.0).astype(jnp.float32)

    # Is agent adjacent to yellow pyramid?
    dist_agent_yp_raw = jnp.abs(agent_y - yp_y) + jnp.abs(agent_x - yp_x)
    agent_adj_yp = (dist_agent_yp_raw <= 1.0).astype(jnp.float32)

    # Is yellow pyramid adjacent to green square? (success)
    dist_yp_gs_raw = jnp.abs(yp_y - gs_y) + jnp.abs(yp_x - gs_x)
    yp_adj_gs = ((dist_yp_gs_raw <= 1.0) & (yp_count > 0)).astype(jnp.float32)

    # Phase indicators
    phase_go_pickup = ((1.0 - holding_yp) * (1.0 - yp_adj_gs))
    phase_carry = holding_yp
    phase_done = yp_adj_gs * (1.0 - holding_yp)

    # Step progress
    step_norm = state.step_num.astype(jnp.float32) / 80.0

    # Find best adjacent cell to green square (nearest free floor cell to agent)
    gs_y_int = gs_y.astype(jnp.int32)
    gs_x_int = gs_x.astype(jnp.int32)

    best_adj_y = gs_y
    best_adj_x = gs_x
    best_adj_dist = H_f + W_f + 1.0

    for d in range(4):
        dy_off = DIRECTIONS[d, 0]
        dx_off = DIRECTIONS[d, 1]
        ny = jnp.clip(gs_y_int + dy_off, 0, H - 1)
        nx = jnp.clip(gs_x_int + dx_off, 0, W - 1)
        is_floor = (state.grid[ny, nx, 0] == FLOOR)
        d_to_adj = jnp.abs(agent_y - ny.astype(jnp.float32)) + jnp.abs(agent_x - nx.astype(jnp.float32))
        better = is_floor & (d_to_adj < best_adj_dist)
        best_adj_y = jnp.where(better, ny.astype(jnp.float32), best_adj_y)
        best_adj_x = jnp.where(better, nx.astype(jnp.float32), best_adj_x)
        best_adj_dist = jnp.where(better, d_to_adj, best_adj_dist)

    norm_best_adj_y = best_adj_y / (H_f - 1)
    norm_best_adj_x = best_adj_x / (W_f - 1)
    rel_agent_best_adj_y = (best_adj_y - agent_y) / H_f
    rel_agent_best_adj_x = (best_adj_x - agent_x) / W_f
    dist_agent_best_adj = (jnp.abs(agent_y - best_adj_y) + jnp.abs(agent_x - best_adj_x)) / (H_f + W_f)

    # All 4 adjacent cells to green square - their distances to agent (for richer signal)
    adj_info = []
    for d in range(4):
        dy_off = DIRECTIONS[d, 0]
        dx_off = DIRECTIONS[d, 1]
        ny = jnp.clip(gs_y_int + dy_off, 0, H - 1)
        nx = jnp.clip(gs_x_int + dx_off, 0, W - 1)
        is_floor = (state.grid[ny, nx, 0] == FLOOR).astype(jnp.float32)
        d_to_adj = (jnp.abs(agent_y - ny.astype(jnp.float32)) + jnp.abs(agent_x - nx.astype(jnp.float32))) / (H_f + W_f)
        adj_info.extend([is_floor, d_to_adj])

    # Local grid view around agent (5x5 neighborhood)
    ay = state.agent.position[0].astype(jnp.int32)
    ax = state.agent.position[1].astype(jnp.int32)
    local_obs = []
    for dy_off in range(-2, 3):
        for dx_off in range(-2, 3):
            ny = jnp.clip(ay + dy_off, 0, H - 1)
            nx = jnp.clip(ax + dx_off, 0, W - 1)
            tid = state.grid[ny, nx, 0].astype(jnp.float32) / 12.0
            tcol = state.grid[ny, nx, 1].astype(jnp.float32) / 11.0
            local_obs.extend([tid, tcol])

    # Additional useful features
    front_dist_to_gs_norm = front_dist_to_gs / (H_f + W_f)
    # Is the agent in a position where PUTDOWN would succeed?
    ready_to_succeed = (holding_yp * front_is_floor * front_adj_to_gs)
    # Facing the green square directly
    facing_gs = front_is_gs

    obs = jnp.array([
        norm_agent_y, norm_agent_x,
        dir_oh[0], dir_oh[1], dir_oh[2], dir_oh[3],
        holding_yp, pocket_empty,
        norm_yp_y, norm_yp_x, yp_on_grid,
        norm_gs_y, norm_gs_x,
        rel_agent_yp_y, rel_agent_yp_x,
        rel_agent_gs_y, rel_agent_gs_x,
        rel_yp_gs_y, rel_yp_gs_x,
        dist_agent_yp, dist_agent_gs, dist_yp_gs,
        front_tile_id, front_tile_color,
        front_is_yp, front_is_gs, front_is_floor,
        front_adj_to_gs, front_ideal_putdown,
        front_dist_to_gs_norm,
        agent_adj_gs, agent_adj_yp,
        yp_adj_gs,
        phase_go_pickup, phase_carry, phase_done,
        step_norm,
        norm_best_adj_y, norm_best_adj_x,
        rel_agent_best_adj_y, rel_agent_best_adj_x,
        dist_agent_best_adj,
        facing_gs,
        ready_to_succeed,
        *adj_info,
        *local_obs,
    ], dtype=jnp.float32)

    return obs


def compute_reward(state, action, next_state):
    H, W = state.grid.shape[:2]
    H_f = jnp.float32(H)
    W_f = jnp.float32(W)
    max_dist = H_f + W_f

    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    # Green square position (stable - doesn't move)
    gs_mask = (state.grid[:, :, 0] == SQUARE) & (state.grid[:, :, 1] == GREEN)
    gs_count = jnp.sum(gs_mask).astype(jnp.float32)
    gs_y = jnp.sum(yy * gs_mask).astype(jnp.float32) / jnp.maximum(gs_count, 1.0)
    gs_x = jnp.sum(xx * gs_mask).astype(jnp.float32) / jnp.maximum(gs_count, 1.0)

    # Yellow pyramid - current state
    yp_mask_cur = (state.grid[:, :, 0] == PYRAMID) & (state.grid[:, :, 1] == YELLOW)
    yp_count_cur = jnp.sum(yp_mask_cur).astype(jnp.float32)
    yp_y_cur = jnp.sum(yy * yp_mask_cur).astype(jnp.float32) / jnp.maximum(yp_count_cur, 1.0)
    yp_x_cur = jnp.sum(xx * yp_mask_cur).astype(jnp.float32) / jnp.maximum(yp_count_cur, 1.0)

    # Yellow pyramid - next state
    yp_mask_next = (next_state.grid[:, :, 0] == PYRAMID) & (next_state.grid[:, :, 1] == YELLOW)
    yp_count_next = jnp.sum(yp_mask_next).astype(jnp.float32)
    yp_y_next = jnp.sum(yy * yp_mask_next).astype(jnp.float32) / jnp.maximum(yp_count_next, 1.0)
    yp_x_next = jnp.sum(xx * yp_mask_next).astype(jnp.float32) / jnp.maximum(yp_count_next, 1.0)

    # Agent positions
    agent_y_cur = state.agent.position[0].astype(jnp.float32)
    agent_x_cur = state.agent.position[1].astype(jnp.float32)
    agent_y_next = next_state.agent.position[0].astype(jnp.float32)
    agent_x_next = next_state.agent.position[1].astype(jnp.float32)

    # Holding states
    holding_cur = (state.agent.pocket[0] == PYRAMID) & (state.agent.pocket[1] == YELLOW)
    holding_next = (next_state.agent.pocket[0] == PYRAMID) & (next_state.agent.pocket[1] == YELLOW)
    just_picked_up = (~holding_cur) & holding_next
    just_put_down = holding_cur & (~holding_next)

    # Success condition: pyramid on grid, adjacent (manhattan<=1) to green square
    dist_yp_gs_cur = jnp.abs(yp_y_cur - gs_y) + jnp.abs(yp_x_cur - gs_x)
    dist_yp_gs_next = jnp.abs(yp_y_next - gs_y) + jnp.abs(yp_x_next - gs_x)

    success_cur = (dist_yp_gs_cur <= 1.0) & (yp_count_cur > 0) & (gs_count > 0)
    success_next = (dist_yp_gs_next <= 1.0) & (yp_count_next > 0) & (gs_count > 0)
    just_succeeded = (~success_cur) & success_next

    # 1. Large success reward
    reward_success = jax.lax.select(just_succeeded, 10.0, 0.0)

    # 2. Milestone: pick up pyramid
    reward_pickup = jax.lax.select(just_picked_up, 2.0, 0.0)

    # 3. Phase 1 shaping: when NOT holding, approach yellow pyramid
    dist_to_yp_cur = jnp.abs(agent_y_cur - yp_y_cur) + jnp.abs(agent_x_cur - yp_x_cur)
    dist_to_yp_next = jnp.abs(agent_y_next - yp_y_cur) + jnp.abs(agent_x_next - yp_x_cur)
    delta_to_yp = (dist_to_yp_cur - dist_to_yp_next) / max_dist
    reward_approach_yp = jnp.where(
        (~holding_cur) & (~success_cur) & (yp_count_cur > 0),
        delta_to_yp * 2.0,
        0.0
    )

    # 4. Phase 2 shaping: when holding, approach green square
    dist_to_gs_cur = jnp.abs(agent_y_cur - gs_y) + jnp.abs(agent_x_cur - gs_x)
    dist_to_gs_next = jnp.abs(agent_y_next - gs_y) + jnp.abs(agent_x_next - gs_x)
    delta_to_gs = (dist_to_gs_cur - dist_to_gs_next) / max_dist
    reward_approach_gs = jnp.where(
        holding_cur & (~success_cur),
        delta_to_gs * 3.0,
        0.0
    )

    # 5. Penalty for putting down pyramid NOT adjacent to green square
    putdown_penalty = jnp.where(
        just_put_down & (~success_next),
        -1.5,
        0.0
    )

    # Bonus for putting down when it achieves success
    putdown_bonus = jnp.where(
        just_put_down & success_next,
        3.0,
        0.0
    )

    # 6. Continuous reward for being in success state
    reward_maintain = jax.lax.select(success_next, 0.2, 0.0)

    # 7. Bonus for becoming adjacent to green square while holding
    agent_adj_gs_cur = (dist_to_gs_cur <= 1.0)
    agent_adj_gs_next = (dist_to_gs_next <= 1.0)
    just_became_adj_gs = holding_cur & (~agent_adj_gs_cur) & agent_adj_gs_next
    reward_adj_gs = jax.lax.select(just_became_adj_gs, 2.0, 0.0)

    # 8. Continuous bonus for holding and being adjacent to green square
    reward_holding_adj = jnp.where(
        holding_next & agent_adj_gs_next & (~success_next),
        0.3,
        0.0
    )

    # 9. Reward for front tile being adjacent to green square when holding
    # This guides the agent to face a good putdown location
    dy_front = DIRECTIONS[next_state.agent.direction, 0]
    dx_front = DIRECTIONS[next_state.agent.direction, 1]
    front_y_next = jnp.clip(next_state.agent.position[0].astype(jnp.int32) + dy_front, 0, H - 1)
    front_x_next = jnp.clip(next_state.agent.position[1].astype(jnp.int32) + dx_front, 0, W - 1)
    front_is_floor_next = (next_state.grid[front_y_next, front_x_next, 0] == FLOOR)
    front_dist_to_gs_next = jnp.abs(front_y_next.astype(jnp.float32) - gs_y) + jnp.abs(front_x_next.astype(jnp.float32) - gs_x)
    front_adj_gs_next = (front_dist_to_gs_next <= 1.0)

    reward_ready_putdown = jnp.where(
        holding_next & front_is_floor_next & front_adj_gs_next & (~success_next),
        0.5,
        0.0
    )

    # 10. Potential-based shaping using pyramid-to-gs distance when on grid
    yp_on_grid_cur = (yp_count_cur > 0) & (~holding_cur)
    yp_on_grid_next = (yp_count_next > 0) & (~holding_next)
    phi_yp_cur = -dist_yp_gs_cur / max_dist
    phi_yp_next = -dist_yp_gs_next / max_dist
    potential_yp_gs = jnp.where(
        yp_on_grid_cur & yp_on_grid_next & (~success_cur),
        (phi_yp_next - phi_yp_cur) * 3.0,
        0.0
    )

    # 11. When holding, reward approaching the best adjacent cell to green square
    gs_y_int = gs_y.astype(jnp.int32)
    gs_x_int = gs_x.astype(jnp.int32)

    best_adj_y = gs_y
    best_adj_x = gs_x
    best_adj_dist_cur = max_dist + 1.0

    for d in range(4):
        dy_off = DIRECTIONS[d, 0]
        dx_off = DIRECTIONS[d, 1]
        ny = jnp.clip(gs_y_int + dy_off, 0, H - 1)
        nx = jnp.clip(gs_x_int + dx_off, 0, W - 1)
        is_floor = (state.grid[ny, nx, 0] == FLOOR)
        d_to_adj = jnp.abs(agent_y_cur - ny.astype(jnp.float32)) + jnp.abs(agent_x_cur - nx.astype(jnp.float32))
        better = is_floor & (d_to_adj < best_adj_dist_cur)
        best_adj_y = jnp.where(better, ny.astype(jnp.float32), best_adj_y)
        best_adj_x = jnp.where(better, nx.astype(jnp.float32), best_adj_x)
        best_adj_dist_cur = jnp.where(better, d_to_adj, best_adj_dist_cur)

    dist_to_best_adj_cur = jnp.abs(agent_y_cur - best_adj_y) + jnp.abs(agent_x_cur - best_adj_x)
    dist_to_best_adj_next = jnp.abs(agent_y_next - best_adj_y) + jnp.abs(agent_x_next - best_adj_x)
    delta_to_best_adj = (dist_to_best_adj_cur - dist_to_best_adj_next) / max_dist

    reward_approach_best_adj = jnp.where(
        holding_cur & (~success_cur) & (~agent_adj_gs_cur),
        delta_to_best_adj * 2.0,
        0.0
    )

    # 12. Extra: when holding and at best adj cell, milestone reward
    at_best_adj_cur = (dist_to_best_adj_cur <= 0.5)
    at_best_adj_next = (dist_to_best_adj_next <= 0.5)
    just_reached_best_adj = holding_cur & (~at_best_adj_cur) & at_best_adj_next
    reward_reached_best_adj = jax.lax.select(just_reached_best_adj, 1.0, 0.0)

    # 13. Strong signal: when holding + adjacent to gs + front is floor adjacent to gs
    # This is the "perfect putdown position" - give strong continuous signal
    reward_perfect_position = jnp.where(
        holding_next & agent_adj_gs_next & front_is_floor_next & front_adj_gs_next & (~success_next),
        0.5,
        0.0
    )

    total = (reward_success + reward_pickup +
             reward_approach_yp + reward_approach_gs +
             putdown_penalty + putdown_bonus +
             reward_maintain + reward_adj_gs + reward_holding_adj +
             reward_ready_putdown + potential_yp_gs +
             reward_approach_best_adj + reward_reached_best_adj +
             reward_perfect_position)

    return total.astype(jnp.float32)
