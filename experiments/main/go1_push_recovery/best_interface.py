"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Stand at the origin and recover from random force impulses applied to the torso. Success = survived full episode AND average position error < 10cm from origin.

Success rate: 32%
Obs dim:      61
Generation:   1
Iteration:    30
Mode:         full
"""

import jax
import jax.numpy as jnp


def get_observation(state) -> jnp.ndarray:
    """
    Observation for Go1 push recovery task.
    Provides comprehensive but well-normalized information.
    """
    # --- IMU / Orientation ---
    gyro = state.info["gyro"]           # (3,) angular velocity in body frame
    gravity = state.info["gravity"]     # (3,) gravity vector in body frame (key for balance)
    upvector = state.info["upvector"]   # (3,) body up direction

    # --- Position relative to origin ---
    pos_xy = state.info["pos_xy"]       # (2,) XY offset from origin
    height = state.info["height"]       # () COM height

    # --- Velocity in body frame ---
    local_linvel = state.info["local_linvel"]  # (3,) linear velocity in body frame

    # --- Heading (encoded as sin/cos to avoid discontinuity) ---
    heading = state.info["heading"]
    heading_sin = jnp.sin(heading)
    heading_cos = jnp.cos(heading)

    # --- Joint states ---
    joint_pos = state.data.qpos[7:]    # (12,) joint angles
    joint_vel = state.data.qvel[6:]    # (12,) joint velocities

    # --- Push force info ---
    push_force = state.info["push_force"]  # (3,)
    push_magnitude = jnp.linalg.norm(push_force)
    push_dir = push_force / (push_magnitude + 1e-6)  # normalized direction
    push_mag_norm = jnp.tanh(push_magnitude / 200.0)  # soft normalization

    # --- Previous action ---
    last_act = state.info["last_act"]   # (12,)

    # --- Computed features ---
    pos_dist = jnp.linalg.norm(pos_xy)

    # Direction to origin in body frame
    cos_h = jnp.cos(-heading)
    sin_h = jnp.sin(-heading)
    to_origin_x = cos_h * (-pos_xy[0]) - sin_h * (-pos_xy[1])
    to_origin_y = sin_h * (-pos_xy[0]) + cos_h * (-pos_xy[1])
    to_origin_body = jnp.array([to_origin_x, to_origin_y])

    # Velocity projected toward origin
    pos_xy_norm_vec = pos_xy / (pos_dist + 1e-6)
    vel_world_approx = local_linvel[:2]
    vel_toward_origin = -jnp.dot(vel_world_approx, pos_xy_norm_vec)

    # Normalize features
    gyro_norm = jnp.tanh(gyro / 5.0)
    pos_xy_norm = jnp.tanh(pos_xy / 0.5)
    height_norm = (height - 0.27) / 0.1
    linvel_norm = jnp.tanh(local_linvel / 2.0)
    joint_vel_norm = jnp.tanh(joint_vel / 10.0)
    pos_dist_norm = jnp.tanh(pos_dist / 0.5)
    to_origin_norm = jnp.tanh(to_origin_body / 0.5)

    obs = jnp.concatenate([
        gravity,                                 # 3 - key for balance (already ~unit)
        gyro_norm,                               # 3 - angular velocity
        upvector,                                # 3 - orientation
        pos_xy_norm,                             # 2 - position from origin
        jnp.array([height_norm]),                # 1 - height
        linvel_norm,                             # 3 - linear velocity
        jnp.array([heading_sin, heading_cos]),   # 2 - heading
        jnp.array([pos_dist_norm]),              # 1 - distance from origin
        to_origin_norm,                          # 2 - direction to origin in body frame
        jnp.array([vel_toward_origin / 2.0]),    # 1 - velocity toward origin
        joint_pos,                               # 12 - joint positions
        joint_vel_norm,                          # 12 - joint velocities
        push_dir,                                # 3 - push force direction
        jnp.array([push_mag_norm]),              # 1 - push magnitude
        last_act,                                # 12 - previous action
    ])  # Total: 3+3+3+2+1+3+2+1+2+1+12+12+3+1+12 = 61

    return obs.astype(jnp.float32)


def compute_reward(state, action, next_state) -> jnp.ndarray:
    """
    Reward for Go1 push recovery.
    
    Key design principles:
    1. Upright is survival - gate other rewards on being upright
    2. Position return is the task - use strong dense signal
    3. Progress reward to help escape local optima
    4. Minimal penalties to avoid conflicting signals
    5. Avoid over-penalizing velocity during recovery
    """
    # Extract from next_state (outcome of action)
    upvector = next_state.info["upvector"]
    pos_xy = next_state.info["pos_xy"]
    heading = next_state.info["heading"]
    gyro = next_state.info["gyro"]
    local_linvel = next_state.info["local_linvel"]

    # Extract from current state for progress
    pos_xy_prev = state.info["pos_xy"]
    last_act = state.info["last_act"]

    # --- Upright score ---
    # Smooth: 1.0 when upright, 0 when tilted/fallen
    upright = jnp.clip(upvector[-1], 0.0, 1.0)
    
    # Use a steeper function to strongly distinguish upright from fallen
    # At upvector[-1]=1.0: reward=1.0; at 0.3: reward~0.09; at 0.0: reward=0.0
    upright_reward = upright ** 2

    # --- Position reward ---
    pos_dist = jnp.linalg.norm(pos_xy)
    # Exponential decay - strong pull toward origin
    # At 0m: 1.0, at 0.1m: ~0.74, at 0.5m: ~0.22
    position_reward = jnp.exp(-2.0 * pos_dist)

    # --- Progress reward: explicitly reward moving toward origin ---
    prev_dist = jnp.linalg.norm(pos_xy_prev)
    progress = prev_dist - pos_dist  # positive = moved closer
    progress_reward = jnp.clip(progress * 20.0, -2.0, 2.0)

    # --- Heading reward ---
    heading_error = jnp.abs(heading)
    heading_error = jnp.minimum(heading_error, 2.0 * jnp.pi - heading_error)
    heading_reward = jnp.exp(-2.0 * heading_error)

    # --- Stability: low angular velocity ---
    angvel_magnitude = jnp.linalg.norm(gyro)
    stability_reward = jnp.exp(-0.3 * angvel_magnitude)

    # --- Small velocity penalty only when near origin (settled behavior) ---
    linvel_magnitude = jnp.linalg.norm(local_linvel)
    near_origin_factor = jnp.exp(-5.0 * pos_dist)  # only active near origin
    vel_penalty = -0.2 * linvel_magnitude * near_origin_factor

    # --- Action smoothness ---
    action_rate = jnp.sum(jnp.square(action - last_act))
    action_penalty = -0.002 * action_rate

    # --- Hard fall penalty ---
    # Smooth approximation: large negative when upvector[-1] < 0.3
    fall_severity = jnp.clip(0.3 - upvector[-1], 0.0, 0.3) / 0.3
    fall_penalty = -3.0 * fall_severity

    # --- Gate position/heading rewards on being upright ---
    # If fallen, position/heading rewards are diminished
    gated_position = upright_reward * position_reward
    gated_heading = upright_reward * heading_reward

    # --- Combined reward ---
    reward = (
        2.0 * upright_reward      # Survival: stay upright
        + 3.0 * gated_position    # Task: be near origin (gated on upright)
        + 1.5 * progress_reward   # Progress: move toward origin
        + 0.5 * gated_heading     # Heading maintenance
        + 0.3 * stability_reward  # Angular stability
        + vel_penalty             # Settle near origin
        + action_penalty          # Smooth control
        + fall_penalty            # Penalize falling
    )

    return reward.astype(jnp.float32)
