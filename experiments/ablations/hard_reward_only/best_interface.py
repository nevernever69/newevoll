"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Pick up blue pyramid (transforms to green ball in hand), then place green ball next to yellow hex. 4-room 13x13 grid with distractors, max_steps=400.
Success rate: 4%
Obs dim:      347
Generation:   3
Iteration:    30
Mode:         reward_only
"""

import jax
import jax.numpy as jnp

def compute_reward(state, action, next_state):
    BALL, PYRAMID, HEX = 3, 5, 11
    BLUE, GREEN, YELLOW = 3, 2, 5

    H, W = next_state.grid.shape[:2]
    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
    max_dist = jnp.float32(H + W)

    prev_pos = state.agent.position.astype(jnp.float32)
    next_pos = next_state.agent.position.astype(jnp.float32)

    # Pocket states
    prev_holding_green = (state.agent.pocket[0] == BALL) & (state.agent.pocket[1] == GREEN)
    next_holding_green = (next_state.agent.pocket[0] == BALL) & (next_state.agent.pocket[1] == GREEN)
    prev_empty = (state.agent.pocket[0] == 0)
    next_empty = (next_state.agent.pocket[0] == 0)

    pickup_event = prev_empty & next_holding_green
    putdown_event = prev_holding_green & next_empty

    # Blue pyramid location - use state (before pickup removes it)
    bp_mask_s = (state.grid[:, :, 0] == PYRAMID) & (state.grid[:, :, 1] == BLUE)
    bp_count_s = jnp.sum(bp_mask_s)
    bp_y_s = jnp.sum(yy * bp_mask_s) / jnp.maximum(bp_count_s, 1)
    bp_x_s = jnp.sum(xx * bp_mask_s) / jnp.maximum(bp_count_s, 1)

    bp_mask_ns = (next_state.grid[:, :, 0] == PYRAMID) & (next_state.grid[:, :, 1] == BLUE)
    bp_count_ns = jnp.sum(bp_mask_ns)
    bp_y_ns = jnp.sum(yy * bp_mask_ns) / jnp.maximum(bp_count_ns, 1)
    bp_x_ns = jnp.sum(xx * bp_mask_ns) / jnp.maximum(bp_count_ns, 1)

    pyramid_y = jnp.where(bp_count_s > 0, bp_y_s, bp_y_ns)
    pyramid_x = jnp.where(bp_count_s > 0, bp_x_s, bp_x_ns)
    pyramid_exists = (bp_count_s > 0) | (bp_count_ns > 0)

    # Yellow hex location
    yh_mask_s = (state.grid[:, :, 0] == HEX) & (state.grid[:, :, 1] == YELLOW)
    yh_mask_ns = (next_state.grid[:, :, 0] == HEX) & (next_state.grid[:, :, 1] == YELLOW)
    yh_count_s = jnp.sum(yh_mask_s)
    yh_count_ns = jnp.sum(yh_mask_ns)
    yh_y_s = jnp.sum(yy * yh_mask_s) / jnp.maximum(yh_count_s, 1)
    yh_x_s = jnp.sum(xx * yh_mask_s) / jnp.maximum(yh_count_s, 1)
    yh_y_ns = jnp.sum(yy * yh_mask_ns) / jnp.maximum(yh_count_ns, 1)
    yh_x_ns = jnp.sum(xx * yh_mask_ns) / jnp.maximum(yh_count_ns, 1)
    hex_y = jnp.where(yh_count_ns > 0, yh_y_ns, yh_y_s)
    hex_x = jnp.where(yh_count_ns > 0, yh_x_ns, yh_x_s)
    hex_exists = (yh_count_ns > 0) | (yh_count_s > 0)

    # Green ball on grid
    gb_mask_ns = (next_state.grid[:, :, 0] == BALL) & (next_state.grid[:, :, 1] == GREEN)
    gb_count_ns = jnp.sum(gb_mask_ns)
    gb_y_ns = jnp.sum(yy * gb_mask_ns) / jnp.maximum(gb_count_ns, 1)
    gb_x_ns = jnp.sum(xx * gb_mask_ns) / jnp.maximum(gb_count_ns, 1)
    dist_gb_hex_ns = jnp.abs(gb_y_ns - hex_y) + jnp.abs(gb_x_ns - hex_x)

    gb_mask_s = (state.grid[:, :, 0] == BALL) & (state.grid[:, :, 1] == GREEN)
    gb_count_s = jnp.sum(gb_mask_s)
    gb_y_s = jnp.sum(yy * gb_mask_s) / jnp.maximum(gb_count_s, 1)
    gb_x_s = jnp.sum(xx * gb_mask_s) / jnp.maximum(gb_count_s, 1)
    dist_gb_hex_s = jnp.abs(gb_y_s - hex_y) + jnp.abs(gb_x_s - hex_x)

    # Task success: green ball on grid adjacent (Manhattan dist <= 1) to yellow hex
    task_success_ns = (gb_count_ns > 0) & hex_exists & (dist_gb_hex_ns <= 1.5)
    task_success_s = (gb_count_s > 0) & hex_exists & (dist_gb_hex_s <= 1.5)
    just_succeeded = task_success_ns & ~task_success_s

    # Distances for shaping
    prev_dist_pyr = jnp.abs(prev_pos[0] - pyramid_y) + jnp.abs(prev_pos[1] - pyramid_x)
    next_dist_pyr = jnp.abs(next_pos[0] - pyramid_y) + jnp.abs(next_pos[1] - pyramid_x)

    prev_dist_hex = jnp.abs(prev_pos[0] - hex_y) + jnp.abs(prev_pos[1] - hex_x)
    next_dist_hex = jnp.abs(next_pos[0] - hex_y) + jnp.abs(next_pos[1] - hex_x)

    # Adjacency to hex
    prev_adj_hex = (prev_dist_hex <= 1.5) & hex_exists
    next_adj_hex = (next_dist_hex <= 1.5) & hex_exists

    # Normalized distance improvements
    delta_pyramid = (prev_dist_pyr - next_dist_pyr) / max_dist
    delta_hex = (prev_dist_hex - next_dist_hex) / max_dist

    # Good/bad putdown
    good_putdown = putdown_event & prev_adj_hex & hex_exists
    bad_putdown = putdown_event & ~prev_adj_hex

    r = jnp.float32(0.0)

    # ===== PHASE 1: Navigate to blue pyramid and pick it up =====
    # Use inverse distance as potential: reward = gamma*phi(s') - phi(s)
    # phi(s) = -dist_to_pyramid / max_dist  (higher when closer)
    phi_pyr_prev = -prev_dist_pyr / max_dist
    phi_pyr_next = -next_dist_pyr / max_dist
    phase1_shaping = jnp.where(
        ~prev_holding_green & pyramid_exists,
        (phi_pyr_next - phi_pyr_prev) * jnp.float32(10.0),
        jnp.float32(0.0)
    )
    r = r + phase1_shaping

    # Large pickup milestone bonus
    r = r + jnp.where(pickup_event, jnp.float32(3.0), jnp.float32(0.0))

    # ===== PHASE 2: Navigate to yellow hex while holding green ball =====
    phi_hex_prev = -prev_dist_hex / max_dist
    phi_hex_next = -next_dist_hex / max_dist
    phase2_shaping = jnp.where(
        prev_holding_green & hex_exists,
        (phi_hex_next - phi_hex_prev) * jnp.float32(10.0),
        jnp.float32(0.0)
    )
    r = r + phase2_shaping

    # Bonus for reaching adjacency to hex while holding
    just_adj_holding = next_adj_hex & ~prev_adj_hex & next_holding_green
    r = r + jnp.where(just_adj_holding, jnp.float32(1.0), jnp.float32(0.0))

    # ===== PHASE 3: Place green ball adjacent to hex =====
    r = r + jnp.where(good_putdown, jnp.float32(5.0), jnp.float32(0.0))
    r = r + jnp.where(bad_putdown, jnp.float32(-2.0), jnp.float32(0.0))

    # After putdown: if ball is on grid, reward closeness to hex (continuous)
    # This helps even if putdown wasn't perfectly adjacent
    ball_placed_shaping = jnp.where(
        (gb_count_ns > 0) & hex_exists & ~task_success_ns,
        -dist_gb_hex_ns / max_dist * jnp.float32(2.0),
        jnp.float32(0.0)
    )
    r = r + ball_placed_shaping

    # ===== TASK COMPLETION =====
    r = r + jnp.where(just_succeeded, jnp.float32(20.0), jnp.float32(0.0))
    r = r + jnp.where(task_success_ns, jnp.float32(1.0), jnp.float32(0.0))

    # Small step penalty for efficiency
    r = r - jnp.float32(0.005)

    return r.astype(jnp.float32)
