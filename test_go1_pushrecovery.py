"""Test script: Go1 Push Recovery.

Novel task: Quadruped stands at origin. Random force impulses are applied to
the torso at regular intervals. The robot must stay upright and return to its
original position after each push.

Success: Survived full episode AND average position error < 10cm from origin.

Design principles (same as PandaPickAndTrack):
  - Few reward terms, all dense, no conflicts.
  - Primary objectives get high scale. Regularization is small.
  - No sparse gates or phase transitions.
  - Position recovery + upright are always active and never conflict.
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
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=500,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        # Push parameters
        push_interval=75,       # steps between pushes (~1.5s)
        push_duration=5,        # steps per push (~0.1s)
        push_force_min=150.0,   # min force magnitude (N)
        push_force_max=400.0,   # max force magnitude (N)
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Core task (dense, high scale)
                position_recovery=8.0,
                heading_recovery=3.0,
                upright=4.0,
                velocity_settle=3.0,
                # Regularization (small)
                termination=-5.0,
                torques=-0.0001,
                action_rate=-0.01,
            ),
        ),
        impl="jax",
        nconmax=4 * 8192,
        njmax=40,
    )


class Go1PushRecovery(go1_base.Go1Env):
    """Stand and recover from random force perturbations."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides=None,
    ):
        super().__init__(
            xml_path=consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._standing_height = float(self._init_q[2])
        # Origin position (XY) to recover to
        self._origin_xy = jp.array(self._init_q[0:2])
        # Original heading (yaw from quaternion)
        self._origin_heading = self._quat_to_yaw(self._init_q[3:7])

    @staticmethod
    def _quat_to_yaw(quat):
        """Extract yaw angle from quaternion [w, x, y, z]."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return jp.arctan2(siny_cosp, cosy_cosp)

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "step_count": jp.array(0),
            "push_force": jp.zeros(3),       # current push force vector
            "cum_pos_error": jp.array(0.0),   # for averaging
            "push_count": jp.array(0.0),      # how many pushes so far
        }

        metrics = {
            f"reward/{k}": jp.zeros(())
            for k in self._config.reward_config.scales.keys()
        }
        metrics["pos_error"] = jp.zeros(())
        metrics["heading_error"] = jp.zeros(())
        metrics["push_count"] = jp.zeros(())
        metrics["success"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        step_count = state.info["step_count"] + 1

        # ── Push logic ──
        interval = self._config.push_interval
        duration = self._config.push_duration
        phase_in_cycle = step_count % interval
        push_active = phase_in_cycle < duration
        at_push_start = phase_in_cycle == 0

        # Generate random force at push start
        push_rng = jax.random.fold_in(state.info["rng"], step_count)
        rng1, rng2 = jax.random.split(push_rng)

        # Random horizontal direction + small vertical component
        angle = jax.random.uniform(rng1, (), minval=0.0, maxval=2.0 * jp.pi)
        force_mag = jax.random.uniform(
            rng2, (),
            minval=self._config.push_force_min,
            maxval=self._config.push_force_max,
        )
        new_force = jp.array([jp.cos(angle), jp.sin(angle), 0.1]) * force_mag

        # Only update force at push start, keep previous during push
        push_force = jp.where(at_push_start, new_force, state.info["push_force"])
        # Zero force when not pushing
        applied_force = jp.where(push_active, push_force, jp.zeros(3))

        # Apply external force to torso
        xfrc = jp.zeros((self.mjx_model.nbody, 6))
        xfrc = xfrc.at[self._torso_body_id, :3].set(applied_force)
        data_with_force = state.data.replace(xfrc_applied=xfrc)

        # Physics step
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, data_with_force, motor_targets, self.n_substeps
        )

        # Track push count
        push_count = state.info["push_count"] + at_push_start.astype(float)

        # Track cumulative position error
        pos_xy = data.qpos[0:2]
        pos_error = jp.linalg.norm(pos_xy - self._origin_xy)
        cum_pos_error = state.info["cum_pos_error"] + pos_error

        info = {
            **state.info,
            "step_count": step_count,
            "push_force": push_force,
            "push_count": push_count,
            "cum_pos_error": cum_pos_error,
            "last_act": action,
        }

        obs = self._get_obs(data, info)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, info, done)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # Success: survived AND avg position error < 10cm
        avg_pos_error = cum_pos_error / jp.maximum(step_count.astype(float), 1.0)
        heading = self._quat_to_yaw(data.qpos[3:7])
        heading_error = jp.abs(heading - self._origin_heading)
        heading_error = jp.minimum(heading_error, 2.0 * jp.pi - heading_error)

        success = (~done.astype(bool)) & (avg_pos_error < 0.10)

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["pos_error"] = pos_error
        state.metrics["heading_error"] = heading_error
        state.metrics["push_count"] = push_count
        state.metrics["success"] = success.astype(float)

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done, info=info)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        up = self.get_upvector(data)
        return up[-1] < 0.3

    def _get_obs(self, data: mjx.Data, info: dict) -> dict:
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        linvel = self.get_local_linvel(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]

        # Position relative to origin
        pos_xy = data.qpos[0:2] - self._origin_xy

        # Heading error
        heading = self._quat_to_yaw(data.qpos[3:7])
        heading_error = heading - self._origin_heading

        state = jp.hstack([
            linvel,                              # 3: local linear velocity
            gyro,                                # 3: angular velocity
            gravity,                             # 3: gravity in body frame
            joint_angles - self._default_pose,   # 12: joint angle offsets
            joint_vel,                           # 12: joint velocities
            info["last_act"],                    # 12: previous action
            pos_xy,                              # 2: XY offset from origin
            jp.array([data.qpos[2]]),            # 1: COM height
            jp.array([heading_error]),           # 1: heading error
            info["push_force"] / 100.0,          # 3: push force (normalized)
        ])

        return {"state": state}

    def _get_reward(self, data, action, info, done) -> dict:
        up = self.get_upvector(data)
        linvel = self.get_local_linvel(data)

        # ── 1. POSITION RECOVERY (primary — stay near origin) ──
        pos_xy = data.qpos[0:2]
        pos_error = jp.linalg.norm(pos_xy - self._origin_xy)
        position_recovery = 1.0 - jp.tanh(5.0 * pos_error)

        # ── 2. HEADING RECOVERY (maintain original heading) ──
        heading = self._quat_to_yaw(data.qpos[3:7])
        heading_error = jp.abs(heading - self._origin_heading)
        heading_error = jp.minimum(heading_error, 2.0 * jp.pi - heading_error)
        heading_recovery = 1.0 - jp.tanh(3.0 * heading_error)

        # ── 3. UPRIGHT (stay upright — critical for survival) ──
        upright = jp.clip(up[-1], 0.0, 1.0)

        # ── 4. VELOCITY SETTLE (penalize lingering velocity — want stillness) ──
        vel_magnitude = jp.linalg.norm(linvel)
        velocity_settle = 1.0 - jp.tanh(2.0 * vel_magnitude)

        # ── Regularization ──
        termination = done.astype(float)
        torques = jp.sum(jp.square(data.actuator_force))
        action_rate = jp.sum(jp.square(action - info["last_act"]))

        return {
            "position_recovery": position_recovery,
            "heading_recovery": heading_recovery,
            "upright": upright,
            "velocity_settle": velocity_settle,
            "termination": termination,
            "torques": torques,
            "action_rate": action_rate,
        }

    @property
    def observation_size(self):
        return {"state": 52}

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu


# ---- Training ----

def main():
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground._src import wrapper

    print("Creating Go1PushRecovery environment...")
    env = Go1PushRecovery()
    print(f"  observation_size: {env.observation_size}")
    print(f"  action_size: {env.action_size}")

    # Sanity check
    print("\nSanity check...")
    rng = jax.random.key(0)
    state = jax.jit(env.reset)(rng)
    print(f"  obs shape: {state.obs['state'].shape}")
    state2 = jax.jit(env.step)(state, jp.zeros(env.action_size))
    print(f"  step reward: {state2.reward}")
    print("  OK")

    num_timesteps = 50_000_000
    print(f"\nTraining for {num_timesteps:,} timesteps...")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="state",
    )

    t0 = time.time()
    metrics_history = []

    def progress(num_steps, metrics):
        elapsed = time.time() - t0
        reward = metrics.get("eval/episode_reward", 0.0)
        pos_err = metrics.get("eval/episode_pos_error", "?")
        heading_err = metrics.get("eval/episode_heading_error", "?")
        pushes = metrics.get("eval/episode_push_count", "?")
        success = metrics.get("eval/episode_success", "?")
        metrics_history.append({"steps": num_steps, "reward": float(reward)})
        print(f"  [{elapsed:6.1f}s] step={num_steps:>12,}  reward={reward:.1f}  pos_err={pos_err}  heading_err={heading_err}  pushes={pushes}  success={success}")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=20,
        reward_scaling=1.0,
        episode_length=500,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=4096,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory=network_factory,
        seed=0,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    eval_env = Go1PushRecovery()

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

    gif_env = Go1PushRecovery()
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
    final_pos_err = float(jp.linalg.norm(state.data.qpos[0:2] - gif_env._origin_xy))
    push_count = float(state.info["push_count"])
    print(f"  Episode reward: {total_reward:.1f}")
    print(f"  Final pos error: {final_pos_err*100:.1f} cm")
    print(f"  Push count: {push_count:.0f}")
    print(f"  Success: {float(state.metrics.get('success', 0))}")

    print("  Rendering frames...")
    frames = gif_env.render(trajectory, height=480, width=640)
    print(f"  Got {len(frames)} frames")

    gif_path = "go1_pushrecovery_trained.gif"
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
