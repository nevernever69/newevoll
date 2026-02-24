"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Place the yellow pyramid adjacent to the green square. Grid 9x9, max_steps=80.
Success rate: 2%
Obs dim:      171
Generation:   1
Iteration:    30
Mode:         reward_only
"""

import jax
import jax.numpy as jnp

def compute_reward(state, action, next_state):
    PYRAMID = 5
    SQUARE = 4
    YELLOW = 5
    GREEN = 2

    H, W = next_state.grid.shape[:2]
    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    def find_obj(grid, tile_id, color_id):
        mask = (grid[:, :, 0] == tile_id) & (grid[:, :, 1] == color_id)
        count = jnp.sum(mask).astype(jnp.float32)
        y = jnp.sum(yy * mask).astype(jnp.float32) / jnp.maximum(count, 1.0)
        x = jnp.sum(xx * mask).astype(jnp.float32) / jnp.maximum(count, 1.0)
        return y, x, count

    # Previous state
    pyr_y_s, pyr_x_s, pyr_cnt_s = find_obj(state.grid, PYRAMID, YELLOW)
    sq_y_s, sq_x_s, sq_cnt_s = find_obj(state.grid, SQUARE, GREEN)
    ay_s = state.agent.position[0].astype(jnp.float32)
    ax_s = state.agent.position[1].astype(jnp.float32)
    holding_s = (state.agent.pocket[0] == PYRAMID) & (state.agent.pocket[1] == YELLOW)

    # Next state
    pyr_y_n, pyr_x_n, pyr_cnt_n = find_obj(next_state.grid, PYRAMID, YELLOW)
    sq_y_n, sq_x_n, sq_cnt_n = find_obj(next_state.grid, SQUARE, GREEN)
    ay_n = next_state.agent.position[0].astype(jnp.float32)
    ax_n = next_state.agent.position[1].astype(jnp.float32)
    holding_n = (next_state.agent.pocket[0] == PYRAMID) & (next_state.agent.pocket[1] == YELLOW)

    hs = holding_s.astype(jnp.float32)
    hn = holding_n.astype(jnp.float32)
    just_picked = (~holding_s) & holding_n
    just_placed = holding_s & (~holding_n)
    pyr_on_grid_n = pyr_cnt_n > 0.5

    max_dist = jnp.float32(H + W)

    # Success condition
    dist_pyr_sq_n = jnp.abs(pyr_y_n - sq_y_n) + jnp.abs(pyr_x_n - sq_x_n)
    success = pyr_on_grid_n & (~holding_n) & (dist_pyr_sq_n >= 0.5) & (dist_pyr_sq_n <= 1.5)

    # Phase 1: not holding -> go to pyramid
    dist_ap_s = jnp.abs(ay_s - pyr_y_s) + jnp.abs(ax_s - pyr_x_s)
    dist_ap_n = jnp.abs(ay_n - pyr_y_n) + jnp.abs(ax_n - pyr_x_n)
    # Reward reduction in distance to pyramid
    r_phase1 = (1.0 - hs) * (1.0 - hn) * (dist_ap_s - dist_ap_n) / max_dist * 2.0

    # Pickup bonus
    r_pickup = just_picked.astype(jnp.float32) * 3.0

    # Phase 2: holding -> go near square
    # Want agent to be close to square (dist == 1 or 2 is ideal for placing adjacent)
    dist_as_s = jnp.abs(ay_s - sq_y_s) + jnp.abs(ax_s - sq_x_s)
    dist_as_n = jnp.abs(ay_n - sq_y_n) + jnp.abs(ax_n - sq_x_n)
    r_phase2 = hs * hn * (dist_as_s - dist_as_n) / max_dist * 2.0

    # Putdown reward: based on where pyramid lands relative to square
    dist_pyr_sq_n_safe = jnp.where(pyr_on_grid_n, dist_pyr_sq_n, max_dist)
    # If placed adjacent: big reward
    placed_adjacent = just_placed & pyr_on_grid_n & (dist_pyr_sq_n_safe >= 0.5) & (dist_pyr_sq_n_safe <= 1.5)
    # If placed close (dist 2-3): small reward
    placed_close = just_placed & pyr_on_grid_n & (dist_pyr_sq_n_safe > 1.5) & (dist_pyr_sq_n_safe <= 3.5)
    # If placed far: penalty
    placed_far = just_placed & pyr_on_grid_n & (dist_pyr_sq_n_safe > 3.5)

    r_putdown = (
        placed_adjacent.astype(jnp.float32) * 10.0
        + placed_close.astype(jnp.float32) * 1.0
        + placed_far.astype(jnp.float32) * (-1.0)
    )

    # Success reward
    r_success = success.astype(jnp.float32) * 20.0

    # Small dense reward: when pyramid is on grid and not held, reward proximity to square
    # This persists across steps to maintain signal
    pyr_placed_n = pyr_on_grid_n & (~holding_n)
    pyr_placed_s = (pyr_cnt_s > 0.5) & (~holding_s)
    dist_pyr_sq_s = jnp.abs(pyr_y_s - sq_y_s) + jnp.abs(pyr_x_s - sq_x_s)
    # Only when pyramid was already placed in previous step too (not just put down)
    r_placement_improvement = (
        pyr_placed_n & pyr_placed_s & (~just_placed)
    ).astype(jnp.float32) * (dist_pyr_sq_s - dist_pyr_sq_n) / max_dist * 1.0

    total = (
        r_phase1
        + r_pickup
        + r_phase2
        + r_putdown
        + r_success
        + r_placement_improvement
    )

    return total.astype(jnp.float32)
