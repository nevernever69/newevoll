"""
Best MDP Interface — evolved by MDP Interface Discovery

Task:         Track a moving 3D target (Lissajous trajectory) with the Panda arm's end-effector. Success = mean tracking error < 2cm over the episode.

Success rate: 67%
Obs dim:      94
Generation:   3
Iteration:    30
Mode:         full
"""

import jax
import jax.numpy as jnp


def get_observation(state) -> jnp.ndarray:
    """
    Observation for Panda tracking task.
    
    Focused design based on what works in the best programs (65%):
    - Robot state (joints, velocities)
    - Task error (multi-scale)
    - Target dynamics (vel, acc, jerk)
    - Trajectory phase and parameters
    - Future target positions (predictive)
    - Control state
    
    Key improvements:
    - Added target jerk (from best programs)
    - More future lookahead points
    - Velocity-error alignment scalar
    - Clipped future errors to [-5, 5]
    """
    qpos = state.data.qpos
    qvel = state.data.qvel

    # === Robot State ===
    arm_qpos = qpos[0:7]
    arm_qpos_norm = arm_qpos / jnp.pi  # ~[-1, 1]

    arm_qvel = qvel[0:7]
    arm_qvel_norm = arm_qvel / 2.0

    # === Task State ===
    gripper_pos = state.info["gripper_pos"]
    target_pos = state.info["target_pos"]
    target_vel = state.info["target_vel"]
    dist = state.info["gripper_target_dist"]

    # Error vector
    error_vec = target_pos - gripper_pos

    # Multi-scale error normalization
    error_fine = error_vec / 0.02
    error_med = error_vec / 0.05
    error_coarse = error_vec / 0.15

    # Scalar distance at multiple scales
    dist_features = jnp.array([
        dist / 0.02,
        dist / 0.05,
        dist / 0.15,
    ])

    # Absolute positions (normalized)
    gripper_pos_norm = gripper_pos / 0.5
    target_pos_norm = target_pos / 0.5

    # === Trajectory Parameters ===
    traj_params = state.info["traj_params"]
    step_count = state.info["step_count"].astype(jnp.float32)
    t = step_count * 0.02

    omega_x = 0.35 * traj_params["freq_x"]
    omega_y = 0.35 * traj_params["freq_y"]
    omega_z = 0.35 * traj_params["freq_z"]

    # Target velocity (normalized)
    target_vel_norm = target_vel / 0.15

    # Target acceleration (analytical second derivative)
    target_acc_x = -0.10 * omega_x * omega_x * jnp.sin(omega_x * t + traj_params["phase_x"])
    target_acc_y = -0.10 * omega_y * omega_y * jnp.sin(omega_y * t + traj_params["phase_y"])
    target_acc_z = -0.07 * omega_z * omega_z * jnp.sin(omega_z * t)
    target_acc = jnp.array([target_acc_x, target_acc_y, target_acc_z]) / 0.03

    # Target jerk (third derivative)
    target_jerk_x = -0.10 * omega_x ** 3 * jnp.cos(omega_x * t + traj_params["phase_x"])
    target_jerk_y = -0.10 * omega_y ** 3 * jnp.cos(omega_y * t + traj_params["phase_y"])
    target_jerk_z = -0.07 * omega_z ** 3 * jnp.cos(omega_z * t)
    target_jerk = jnp.array([target_jerk_x, target_jerk_y, target_jerk_z]) / 0.05

    # === Control State ===
    prev_ctrl = state.info["prev_ctrl"][:7]
    prev_ctrl_norm = prev_ctrl / jnp.pi

    # Control-to-joint error
    ctrl_joint_error = (prev_ctrl - arm_qpos) / 0.3

    # === Trajectory Phase Encoding ===
    phase_arg_x = omega_x * t + traj_params["phase_x"]
    phase_arg_y = omega_y * t + traj_params["phase_y"]
    phase_arg_z = omega_z * t

    traj_phase_x = jnp.array([jnp.sin(phase_arg_x), jnp.cos(phase_arg_x)])
    traj_phase_y = jnp.array([jnp.sin(phase_arg_y), jnp.cos(phase_arg_y)])
    traj_phase_z = jnp.array([jnp.sin(phase_arg_z), jnp.cos(phase_arg_z)])

    # Initial phase (constant per episode)
    init_phase_x = jnp.array([jnp.sin(traj_params["phase_x"]), jnp.cos(traj_params["phase_x"])])
    init_phase_y = jnp.array([jnp.sin(traj_params["phase_y"]), jnp.cos(traj_params["phase_y"])])

    # Normalized frequencies
    freq_features = jnp.array([
        traj_params["freq_x"] / 3.0,
        traj_params["freq_y"] / 3.0,
        traj_params["freq_z"] / 2.0,
    ])

    # === Predictive Features ===
    def future_target_pos(dt):
        tf = t + dt
        fx = 0.45 + 0.10 * jnp.sin(omega_x * tf + traj_params["phase_x"])
        fy = 0.00 + 0.10 * jnp.sin(omega_y * tf + traj_params["phase_y"])
        fz = 0.18 + 0.07 * jnp.sin(omega_z * tf)
        return jnp.array([fx, fy, fz])

    f1 = future_target_pos(0.04)   # 2 steps ahead
    f2 = future_target_pos(0.10)   # 5 steps ahead
    f3 = future_target_pos(0.20)   # 10 steps ahead
    f4 = future_target_pos(0.40)   # 20 steps ahead
    f5 = future_target_pos(0.80)   # 40 steps ahead
    f6 = future_target_pos(1.60)   # 80 steps ahead

    # Future errors relative to gripper (clipped)
    future_err_1 = jnp.clip((f1 - gripper_pos) / 0.02, -5.0, 5.0)
    future_err_2 = jnp.clip((f2 - gripper_pos) / 0.03, -5.0, 5.0)
    future_err_3 = jnp.clip((f3 - gripper_pos) / 0.05, -5.0, 5.0)
    future_err_4 = jnp.clip((f4 - gripper_pos) / 0.08, -5.0, 5.0)
    future_err_5 = jnp.clip((f5 - gripper_pos) / 0.12, -5.0, 5.0)
    future_err_6 = jnp.clip((f6 - gripper_pos) / 0.18, -5.0, 5.0)

    # Future absolute positions (normalized)
    future_pos_1 = f2 / 0.5   # 0.1s ahead
    future_pos_2 = f4 / 0.5   # 0.4s ahead

    # === Time Feature ===
    t_norm = jnp.array([step_count / 500.0])

    # === Velocity-error alignment ===
    error_dir = error_vec / (jnp.linalg.norm(error_vec) + 1e-6)
    vel_error_alignment = jnp.dot(target_vel, error_dir)
    vel_error_alignment_arr = jnp.array([vel_error_alignment / 0.15])

    obs = jnp.concatenate([
        arm_qpos_norm,           # 7: joint positions
        arm_qvel_norm,           # 7: joint velocities
        gripper_pos_norm,        # 3: gripper position
        target_pos_norm,         # 3: target position
        error_fine,              # 3: error / 2cm
        error_med,               # 3: error / 5cm
        error_coarse,            # 3: error / 15cm
        dist_features,           # 3: distance at 3 scales
        target_vel_norm,         # 3: target velocity
        target_acc,              # 3: target acceleration
        target_jerk,             # 3: target jerk
        prev_ctrl_norm,          # 7: previous control
        ctrl_joint_error,        # 7: ctrl-joint error
        freq_features,           # 3: trajectory frequencies
        init_phase_x,            # 2: initial phase x
        init_phase_y,            # 2: initial phase y
        traj_phase_x,            # 2: current phase x
        traj_phase_y,            # 2: current phase y
        traj_phase_z,            # 2: current phase z
        future_err_1,            # 3: future error 0.04s
        future_err_2,            # 3: future error 0.10s
        future_err_3,            # 3: future error 0.20s
        future_err_4,            # 3: future error 0.40s
        future_err_5,            # 3: future error 0.80s
        future_err_6,            # 3: future error 1.60s
        future_pos_1,            # 3: future target 0.1s
        future_pos_2,            # 3: future target 0.4s
        vel_error_alignment_arr, # 1: velocity-error alignment
        t_norm,                  # 1: normalized time
    ]).astype(jnp.float32)

    return obs


def compute_reward(state, action, next_state) -> jnp.ndarray:
    """
    Reward for Panda tracking task.
    
    Success = mean tracking error < 2cm over all 500 steps.
    
    Design: Clean, focused reward signal
    - Multi-scale Gaussian primary reward (proven to work)
    - Mild smoothness penalty to encourage stable tracking
    - No conflicting signals
    
    Key insight from best programs (65%):
    - Multi-scale Gaussian with weights [0.6, 0.3, 0.1] for [2cm, 5cm, 15cm]
    - Small velocity alignment bonus
    - Mild control change penalty
    - No discrete bonuses that create non-smooth gradients
    """
    dist_next = next_state.info["gripper_target_dist"]

    # === Primary Tracking Reward ===
    # Multi-scale Gaussian: strong near threshold, guidance from far

    # Tight: rewards precision within 2cm (success threshold)
    tight_reward = jnp.exp(-0.5 * (dist_next / 0.02) ** 2)

    # Medium: guides from moderate distances
    medium_reward = jnp.exp(-0.5 * (dist_next / 0.05) ** 2)

    # Coarse: helps when far away
    coarse_reward = jnp.exp(-0.5 * (dist_next / 0.15) ** 2)

    # Weighted combination: emphasize tight tracking
    primary_reward = 0.6 * tight_reward + 0.3 * medium_reward + 0.1 * coarse_reward

    # === Velocity Alignment Bonus ===
    # Reward gripper for moving with the target
    error_vec = next_state.info["target_pos"] - next_state.info["gripper_pos"]
    target_vel = next_state.info["target_vel"]

    error_norm = jnp.linalg.norm(error_vec) + 1e-6
    error_dir = error_vec / error_norm

    target_vel_mag = jnp.linalg.norm(target_vel) + 1e-6
    target_vel_dir = target_vel / target_vel_mag

    # Alignment between error direction and target velocity direction
    vel_alignment = jnp.dot(error_dir, target_vel_dir)

    # Scale by distance: more important when not at target
    dist_weight = jnp.clip(dist_next / 0.05, 0.0, 1.0)
    vel_bonus = 0.05 * vel_alignment * dist_weight

    # === Smoothness Penalty ===
    # Penalize large control changes (jerk)
    ctrl_change = next_state.data.ctrl[:7] - state.info["prev_ctrl"][:7]
    ctrl_change_penalty = -0.002 * jnp.linalg.norm(ctrl_change)

    # Small action penalty
    action_penalty = -0.001 * jnp.linalg.norm(action[:7])

    # === Total Reward ===
    reward = (
        primary_reward
        + vel_bonus
        + ctrl_change_penalty
        + action_penalty
    )

    return reward.astype(jnp.float32)
