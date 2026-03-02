"""Test script: Go1 bounce-spin.

Novel task: The quadruped must repeatedly jump (bounce) while rotating (spinning)
around its yaw axis. All body-local sensors — no external targets needed.

Success: COM exceeds height threshold N times AND accumulated yaw > X degrees AND upright.

Design principles (same as Panda tracking):
  - Few reward terms, all dense, no conflicts.
  - Primary objectives get high scale. Regularization is small.
  - No sparse gates or phase transitions.
  - bounce_height + feet_lift create smooth gradient toward jumping.
  - yaw_rate is always active — spinning on ground bootstraps, spinning in air is better.
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
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Core task (dense, high scale)
                bounce_height=10.0,
                yaw_rate=5.0,
                upright=3.0,
                feet_lift=4.0,
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


class Go1BounceSpin(go1_base.Go1Env):
    """Bounce while spinning around yaw axis."""

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

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Small random perturbation
        dxy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

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
            "total_yaw": jp.array(0.0),
            "bounce_count": jp.array(0.0),
            "was_airborne": jp.array(0.0),
            "peak_height": jp.array(0.0),
        }

        metrics = {
            f"reward/{k}": jp.zeros(())
            for k in self._config.reward_config.scales.keys()
        }
        metrics["total_yaw_deg"] = jp.zeros(())
        metrics["bounce_count"] = jp.zeros(())
        metrics["success"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        # Detect contact per foot
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._feet_floor_found_sensor
        ])
        all_airborne = jp.sum(contact) == 0
        all_grounded = jp.sum(contact) >= 2

        # Track bounces
        just_landed = all_grounded & (state.info["was_airborne"] > 0.5)
        was_high = state.info["peak_height"] > self._standing_height + 0.03
        valid_bounce = just_landed & was_high
        bounce_count = state.info["bounce_count"] + valid_bounce.astype(float)

        com_z = data.qpos[2]
        peak_height = jp.where(
            all_airborne,
            jp.maximum(state.info["peak_height"], com_z),
            jp.where(just_landed, jp.array(0.0), state.info["peak_height"]),
        )

        # Track yaw rotation
        gyro = self.get_gyro(data)
        yaw_rate = gyro[2]
        total_yaw = state.info["total_yaw"] + jp.abs(yaw_rate) * self.dt

        info = {
            **state.info,
            "was_airborne": all_airborne.astype(float),
            "bounce_count": bounce_count,
            "peak_height": peak_height,
            "total_yaw": total_yaw,
            "last_act": action,
        }

        obs = self._get_obs(data, info)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, info, done)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        total_yaw_deg = total_yaw * 180.0 / jp.pi
        success = (
            (bounce_count >= 3.0) & (total_yaw_deg >= 180.0) & (~done.astype(bool))
        )

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["total_yaw_deg"] = total_yaw_deg
        state.metrics["bounce_count"] = bounce_count
        state.metrics["success"] = success.astype(float)

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        up = self.get_upvector(data)
        return up[-1] < 0.3

    def _get_obs(self, data: mjx.Data, info: dict) -> dict:
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        linvel = self.get_local_linvel(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]

        state = jp.hstack([
            linvel,                              # 3
            gyro,                                # 3
            gravity,                             # 3
            joint_angles - self._default_pose,   # 12
            joint_vel,                           # 12
            info["last_act"],                    # 12
            jp.array([data.qpos[2]]),            # 1: COM height
            jp.array([info["total_yaw"]]),       # 1
            jp.array([info["bounce_count"]]),    # 1
            jp.array([info["was_airborne"]]),     # 1
        ])

        return {"state": state}

    def _get_reward(self, data, action, info, done) -> dict:
        gyro = self.get_gyro(data)
        up = self.get_upvector(data)
        com_z = data.qpos[2]

        # ── 1. BOUNCE HEIGHT (primary — reward being above standing) ──
        # Gentler tanh for wider gradient. Even small hops get reward.
        height_above = jp.maximum(com_z - self._standing_height, 0.0)
        bounce_height = jp.tanh(3.0 * height_above)

        # ── 2. YAW RATE (primary — reward spinning WHILE airborne) ──
        # Scale by height so spinning on ground gives little reward.
        # Spinning while airborne is what we actually want.
        height_scale = jp.clip(height_above / 0.05, 0.0, 1.0)  # ramps 0→1 over 5cm
        yaw_rate = jp.tanh(jp.abs(gyro[2]) / 3.0) * (0.1 + 0.9 * height_scale)

        # ── 3. UPRIGHT (safety — stay upright) ──
        upright = jp.clip(up[-1], 0.0, 1.0)

        # ── 4. FEET LIFT (smooth — reward feet being off ground) ──
        # Dense gradient toward lifting feet, unlike binary "all airborne".
        # Even lifting one foot slightly gets partial reward.
        feet_z = data.site_xpos[self._feet_site_id, 2]
        avg_feet_height = jp.mean(jp.clip(feet_z, 0.0, 0.1))
        feet_lift = jp.tanh(20.0 * avg_feet_height)

        # ── Regularization ──
        termination = done.astype(float)
        torques = jp.sum(jp.square(data.actuator_force))
        action_rate = jp.sum(jp.square(action - info["last_act"]))

        return {
            "bounce_height": bounce_height,
            "yaw_rate": yaw_rate,
            "upright": upright,
            "feet_lift": feet_lift,
            "termination": termination,
            "torques": torques,
            "action_rate": action_rate,
        }

    @property
    def observation_size(self):
        return {"state": 49}

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu


# ---- Training ----

def main():
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground._src import wrapper

    print("Creating Go1BounceSpin environment...")
    env = Go1BounceSpin()
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
        bounces = metrics.get("eval/episode_bounce_count", "?")
        yaw = metrics.get("eval/episode_total_yaw_deg", "?")
        success = metrics.get("eval/episode_success", "?")
        metrics_history.append({"steps": num_steps, "reward": float(reward)})
        print(f"  [{elapsed:6.1f}s] step={num_steps:>12,}  reward={reward:.1f}  bounces={bounces}  yaw_deg={yaw}  success={success}")

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

    eval_env = Go1BounceSpin()

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


if __name__ == "__main__":
    main()
