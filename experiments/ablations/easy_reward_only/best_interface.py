"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Pick up the blue pyramid. Grid 9x9, max_steps=80.
Success rate: 59%
Obs dim:      171
Generation:   1
Iteration:    30
Mode:         reward_only
"""

import jax
import jax.numpy as jnp

def compute_reward(state, action, next_state):
    PYRAMID = 5
    BLUE = 3
    PICKUP_ACTION = 3

    H, W = state.grid.shape[:2]

    # Pocket state checks
    holding_next = (
        (next_state.agent.pocket[0] == PYRAMID) &
        (next_state.agent.pocket[1] == BLUE)
    )
    holding_curr = (
        (state.agent.pocket[0] == PYRAMID) &
        (state.agent.pocket[1] == BLUE)
    )
    pocket_empty_curr = (state.agent.pocket[0] == 0)

    just_picked_up = holding_next & (~holding_curr)

    # Find blue pyramid on the grid (current state)
    ys = jnp.arange(H)
    xs = jnp.arange(W)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    pyramid_mask_curr = (
        (state.grid[:, :, 0] == PYRAMID) &
        (state.grid[:, :, 1] == BLUE)
    )

    count_curr = jnp.sum(pyramid_mask_curr).astype(jnp.float32)
    pyramid_on_grid = count_curr > 0

    pyr_y = jnp.sum(yy * pyramid_mask_curr).astype(jnp.float32) / jnp.maximum(count_curr, 1.0)
    pyr_x = jnp.sum(xx * pyramid_mask_curr).astype(jnp.float32) / jnp.maximum(count_curr, 1.0)

    # Agent positions
    agent_y_curr = state.agent.position[0].astype(jnp.float32)
    agent_x_curr = state.agent.position[1].astype(jnp.float32)
    agent_y_next = next_state.agent.position[0].astype(jnp.float32)
    agent_x_next = next_state.agent.position[1].astype(jnp.float32)

    dist_curr = jnp.abs(agent_y_curr - pyr_y) + jnp.abs(agent_x_curr - pyr_x)
    dist_next = jnp.abs(agent_y_next - pyr_y) + jnp.abs(agent_x_next - pyr_x)

    max_dist = jnp.float32(H + W - 2)

    # Potential-based distance shaping (only when pyramid is on grid and not holding)
    dist_shaping = jnp.where(
        pyramid_on_grid & (~holding_curr),
        (dist_curr - dist_next) / max_dist * jnp.float32(10.0),
        jnp.float32(0.0)
    )

    # Facing direction checks
    DIRECTIONS = jnp.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=jnp.int32)

    dy_next = DIRECTIONS[next_state.agent.direction, 0]
    dx_next = DIRECTIONS[next_state.agent.direction, 1]
    front_y_next = jnp.clip(next_state.agent.position[0].astype(jnp.int32) + dy_next, 0, H - 1)
    front_x_next = jnp.clip(next_state.agent.position[1].astype(jnp.int32) + dx_next, 0, W - 1)

    front_is_pyramid_next = (
        (next_state.grid[front_y_next, front_x_next, 0] == PYRAMID) &
        (next_state.grid[front_y_next, front_x_next, 1] == BLUE)
    )

    dy_curr = DIRECTIONS[state.agent.direction, 0]
    dx_curr = DIRECTIONS[state.agent.direction, 1]
    front_y_curr = jnp.clip(state.agent.position[0].astype(jnp.int32) + dy_curr, 0, H - 1)
    front_x_curr = jnp.clip(state.agent.position[1].astype(jnp.int32) + dx_curr, 0, W - 1)

    was_facing_pyramid = (
        (state.grid[front_y_curr, front_x_curr, 0] == PYRAMID) &
        (state.grid[front_y_curr, front_x_curr, 1] == BLUE)
    )

    # Milestone: newly facing the pyramid
    just_facing = front_is_pyramid_next & (~was_facing_pyramid) & (~holding_curr)
    facing_milestone = jnp.where(just_facing, jnp.float32(3.0), jnp.float32(0.0))

    # Persistent reward for facing pyramid (encourages maintaining alignment for pickup)
    persistent_facing = jnp.where(
        front_is_pyramid_next & (~holding_next),
        jnp.float32(0.5),
        jnp.float32(0.0)
    )

    # Strong bonus for attempting pickup while facing pyramid with empty pocket
    pickup_bonus = jnp.where(
        (action == PICKUP_ACTION) & was_facing_pyramid & pocket_empty_curr & (~holding_curr),
        jnp.float32(8.0),
        jnp.float32(0.0)
    )

    # Large success reward
    success_reward = jnp.where(just_picked_up, jnp.float32(30.0), jnp.float32(0.0))

    # Step penalty only while not holding (encourages efficiency)
    step_penalty = jnp.where(
        ~holding_next,
        jnp.float32(-0.05),
        jnp.float32(0.0)
    )

    # Additional reward: inverse distance bonus (absolute, not delta) to help with sparse exploration
    # This gives a continuous signal even when not moving
    inv_dist_bonus = jnp.where(
        pyramid_on_grid & (~holding_curr),
        (max_dist - dist_curr) / max_dist * jnp.float32(0.1),
        jnp.float32(0.0)
    )

    total_reward = (
        success_reward
        + dist_shaping
        + facing_milestone
        + persistent_facing
        + pickup_bonus
        + step_penalty
        + inv_dist_bonus
    )
    return total_reward.astype(jnp.float32)
