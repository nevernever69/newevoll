"""Test script: Panda Trajectory Tracking.

Control task: The Panda arm must track a moving 3D target (Lissajous curve)
with its end-effector as precisely as possible.

Single-phase task — pure tracking, no grasping or manipulation.

The target follows a randomized Lissajous trajectory with varying frequencies,
phases, and a steeper vertical component for added difficulty.

Industrially relevant: weld seam tracing, surface inspection,
precision painting, laser cutting paths.

Success: mean tracking error < 2cm over the episode.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import functools
import sys
import time

sys.path.insert(0, "mujoco_playground")

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State
from mujoco_playground._src.manipulation.franka_emika_panda import panda


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=500,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                gripper_target=8.0,
                no_floor_collision=0.25,
                robot_target_qpos=0.3,
                action_smoothness=0.1,
            )
        ),
        # Trajectory parameters — steeper curves for difficulty
        target_speed=0.35,
        target_radius=0.10,
        target_center_x=0.45,
        target_center_y=0.0,
        target_center_z=0.18,
        impl="jax",
        nconmax=12 * 8192,
        njmax=44,
    )


class PandaTracking(panda.PandaBase):
    """Track a moving 3D Lissajous target with the end-effector."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides=None,
    ):
        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "franka_emika_panda"
            / "xmls"
            / "mjx_single_cube.xml"
        )
        super().__init__(xml_path, config, config_overrides)
        self._post_init(obj_name="box", keyframe="home")

        self._floor_hand_found_sensor = [
            self._mj_model.sensor(f"{geom}_floor_found").id
            for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
        ]

    def _generate_trajectory_params(self, rng):
        """Generate randomized Lissajous curve parameters (harder ranges)."""
        rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)
        freq_x = jax.random.uniform(rng1, minval=1.0, maxval=3.0)
        freq_y = jax.random.uniform(rng2, minval=1.0, maxval=3.0)
        freq_z = jax.random.uniform(rng3, minval=0.8, maxval=2.0)
        phase_x = jax.random.uniform(rng4, minval=0.0, maxval=2 * jp.pi)
        phase_y = jax.random.uniform(rng5, minval=0.0, maxval=2 * jp.pi)
        return {
            "freq_x": freq_x,
            "freq_y": freq_y,
            "freq_z": freq_z,
            "phase_x": phase_x,
            "phase_y": phase_y,
        }

    def _get_target_pos(self, t, traj_params):
        """Compute target position at time t from Lissajous parameters."""
        speed = self._config.target_speed
        radius = self._config.target_radius
        x = self._config.target_center_x + radius * jp.sin(
            speed * traj_params["freq_x"] * t + traj_params["phase_x"]
        )
        y = self._config.target_center_y + radius * jp.sin(
            speed * traj_params["freq_y"] * t + traj_params["phase_y"]
        )
        z = self._config.target_center_z + 0.7 * radius * jp.sin(
            speed * traj_params["freq_z"] * t
        )
        return jp.array([x, y, z])

    def _get_target_vel(self, t, traj_params):
        """Compute target velocity at time t (analytical derivative)."""
        speed = self._config.target_speed
        radius = self._config.target_radius
        vx = radius * speed * traj_params["freq_x"] * jp.cos(
            speed * traj_params["freq_x"] * t + traj_params["phase_x"]
        )
        vy = radius * speed * traj_params["freq_y"] * jp.cos(
            speed * traj_params["freq_y"] * t + traj_params["phase_y"]
        )
        vz = 0.7 * radius * speed * traj_params["freq_z"] * jp.cos(
            speed * traj_params["freq_z"] * t
        )
        return jp.array([vx, vy, vz])

    def reset(self, rng: jax.Array) -> State:
        rng, rng_traj = jax.random.split(rng, 2)

        traj_params = self._generate_trajectory_params(rng_traj)

        # Standard home position, no box manipulation needed
        init_q = jp.array(self._init_q)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # Place mocap target at trajectory start for visualization
        target_pos = self._get_target_pos(0.0, traj_params)
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        )

        metrics = {
            "out_of_bounds": jp.array(0.0),
            "tracking_dist": jp.array(0.0),
            "success": jp.array(0.0),
            **{k: jp.array(0.0) for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "traj_params": traj_params,
            "step_count": jp.array(0.0),
            "prev_ctrl": self._init_ctrl,
            "cumulative_tracking_error": jp.array(0.0),
            "tracking_steps": jp.array(0.0),
        }
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        step_count = state.info["step_count"] + 1
        t = step_count * self._config.ctrl_dt

        # Update trajectory target
        traj_params = state.info["traj_params"]
        target_pos = self._get_target_pos(t, traj_params)
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        )

        # Track gripper-to-target error
        gripper_pos = data.site_xpos[self._gripper_site]
        tracking_dist = jp.linalg.norm(gripper_pos - target_pos)
        cumulative_tracking_error = (
            state.info["cumulative_tracking_error"] + tracking_dist
        )
        tracking_steps = state.info["tracking_steps"] + 1.0

        info = {
            **state.info,
            "step_count": step_count,
            "prev_ctrl": state.data.ctrl,
            "cumulative_tracking_error": cumulative_tracking_error,
            "tracking_steps": tracking_steps,
        }

        raw_rewards = self._get_reward(data, info, target_pos, ctrl)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in raw_rewards.items()
        }
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        # Success: mean gripper-to-target error < 2cm
        mean_tracking = cumulative_tracking_error / jp.maximum(tracking_steps, 1.0)
        success = (mean_tracking < 0.02).astype(float)

        state.metrics.update(
            **raw_rewards,
            out_of_bounds=jp.array(0.0),
            tracking_dist=tracking_dist,
            success=success,
        )

        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, state.metrics, info)

    def _get_reward(self, data, info, target_pos, ctrl):
        """Tracking reward: minimize gripper-to-target distance."""
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_target_dist = jp.linalg.norm(gripper_pos - target_pos)

        # 1. GRIPPER → TARGET (main tracking reward)
        gripper_target = 1 - jp.tanh(5.0 * gripper_target_dist)

        # Regularization
        hand_floor_collision = [
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._floor_hand_found_sensor
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision = (1 - floor_collision).astype(float)

        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Action smoothness (penalize jerk)
        action_smoothness = 1 - jp.tanh(
            jp.linalg.norm(ctrl - info["prev_ctrl"]) * 5.0
        )

        return {
            "gripper_target": gripper_target,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
            "action_smoothness": action_smoothness,
        }

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        gripper_pos = data.site_xpos[self._gripper_site]
        t = info["step_count"] * self._config.ctrl_dt
        traj_params = info["traj_params"]

        target_pos = self._get_target_pos(t, traj_params)
        target_vel = self._get_target_vel(t, traj_params)

        # Lookahead targets for anticipatory control
        target_future_1 = self._get_target_pos(t + 0.2, traj_params)
        target_future_2 = self._get_target_pos(t + 0.5, traj_params)
        target_future_3 = self._get_target_pos(t + 1.0, traj_params)

        return jp.concatenate([
            data.qpos[self._robot_arm_qposadr],     # 7: arm joint positions
            data.qvel[self._robot_arm_qposadr],      # 7: arm joint velocities
            gripper_pos,                              # 3: end-effector position
            target_pos,                               # 3: current target
            target_vel,                               # 3: target velocity
            target_pos - gripper_pos,                 # 3: tracking error vector
            target_future_1 - gripper_pos,            # 3: future error 0.2s
            target_future_2 - gripper_pos,            # 3: future error 0.5s
            target_future_3 - gripper_pos,            # 3: future error 1.0s
            info["prev_ctrl"][:7],                    # 7: previous arm control
        ])


# ---- Training ----

def main():
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground._src import wrapper

    print("Creating PandaTracking environment...")
    env = PandaTracking()
    print(f"  observation_size: {env.observation_size}")
    print(f"  action_size: {env.action_size}")

    # Quick sanity check
    print("\nSanity check...")
    rng = jax.random.key(0)
    state = jax.jit(env.reset)(rng)
    print(f"  obs shape: {state.obs.shape}")
    state2 = jax.jit(env.step)(state, jp.zeros(env.action_size))
    print(f"  step reward: {state2.reward}")
    print(f"  tracking_dist: {state2.metrics['tracking_dist']}")
    print("  OK")

    num_timesteps = 50_000_000
    print(f"\nTraining for {num_timesteps:,} timesteps...")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32, 32, 32, 32),
        value_hidden_layer_sizes=(256, 256, 256, 256, 256),
        policy_obs_key="state",
        value_obs_key="state",
    )

    t0 = time.time()
    metrics_history = []

    def progress(num_steps, metrics):
        elapsed = time.time() - t0
        reward = metrics.get("eval/episode_reward", 0.0)
        success = metrics.get("eval/episode_success", "?")
        tracking = metrics.get("eval/episode_tracking_dist", "?")
        metrics_history.append({"steps": num_steps, "reward": float(reward)})
        print(f"  [{elapsed:6.1f}s] step={num_steps:>12,}  reward={reward:.1f}  success={success}  tracking={tracking}")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=20,
        reward_scaling=1.0,
        episode_length=500,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=5e-4,
        entropy_cost=1.5e-2,
        num_envs=2048,
        batch_size=1024,
        max_grad_norm=1.0,
        network_factory=network_factory,
        seed=0,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    eval_env = PandaTracking()

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if metrics_history:
        print(f"Final eval reward: {metrics_history[-1]['reward']:.1f}")
        print(f"Best eval reward:  {max(m['reward'] for m in metrics_history):.1f}")

    # ---- Render GIF ----
    print("\n--- Rendering GIF ---")
    from PIL import Image

    gif_env = PandaTracking()
    inference_fn = make_inference_fn(params)
    jit_inference = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)
    state = gif_env.reset(reset_rng)
    trajectory = [state]

    @jax.jit
    def gif_step(state, action):
        return gif_env.step(state, action)

    num_gif_steps = 500
    print(f"  Rolling out {num_gif_steps} steps...")
    for i in range(num_gif_steps):
        rng, action_rng = jax.random.split(rng)
        action, _ = jit_inference(state.obs, action_rng)
        state = gif_step(state, action)
        trajectory.append(state)
        if state.done:
            print(f"  Episode ended at step {i+1}")
            break

    total_reward = sum(float(s.reward) for s in trajectory[1:])
    tracking_steps = float(state.info["tracking_steps"])
    mean_tracking = float(state.info["cumulative_tracking_error"] / max(tracking_steps, 1))
    print(f"  Episode reward: {total_reward:.1f}")
    print(f"  Mean tracking error: {mean_tracking*100:.1f} cm")
    print(f"  Steps: {tracking_steps:.0f}/{num_gif_steps}")
    print(f"  Success (< 2cm): {float(state.metrics.get('success', 0))}")

    print("  Rendering frames...")
    frames = gif_env.render(trajectory, height=480, width=640)
    print(f"  Got {len(frames)} frames")

    gif_path = "panda_tracking_trained.gif"
    images = [Image.fromarray(f) for f in frames[::4]]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=40,
        loop=0,
    )
    file_size = os.path.getsize(gif_path) / 1024
    print(f"  Saved: {gif_path} ({len(images)} frames, {file_size:.0f}KB)")


if __name__ == "__main__":
    main()
