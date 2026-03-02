"""Test script: Panda pick cube and place on table target.

Novel task: Pick up cube, carry it, place it down on a target XY mark on the table.
Unlike existing PandaPickCube (lifts to mid-air target), this requires placing ON the surface.

Success: cube within 3cm of target XY AND cube z < 0.06 (on table) AND gripper open
         AND cube was lifted at least 15cm above table at some point.

Design principles:
  - Robot MUST lift cube >= 15cm above table before transport/place/release unlock.
  - Sticky "has_lifted" flag ONLY gates later phases (transport, place, release, success).
    Reach and grasp are ALWAYS active — robot can recover from drops.
  - No lift/lower conflict: lift reward is only active BEFORE has_lifted is set.
    Once lifted, transport+place take over (no competing height reward).
  - Place reward gated on XY proximity to target — only rewards lowering when near target.
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
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State
from mujoco_playground._src.manipulation.franka_emika_panda import panda

# Minimum height the cube must reach above its starting z before placement unlocks
LIFT_HEIGHT = 0.08  # 8 cm


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=300,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                reach=1.5,
                grasp=2.0,
                lift=6.0,
                transport=8.0,
                place=14.0,
                release=10.0,
                success_bonus=5.0,
                no_floor_collision=0.25,
                robot_target_qpos=0.15,
            )
        ),
        impl="jax",
        nconmax=12 * 8192,
        njmax=44,
    )


class PandaPlaceCube(panda.PandaBase):
    """Pick up cube and place it on a target location on the table surface."""

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

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_target = jax.random.split(rng, 3)

        # Box starts on table at random position
        box_pos = (
            jax.random.uniform(
                rng_box,
                (3,),
                minval=jp.array([-0.15, -0.15, 0.0]),
                maxval=jp.array([0.15, 0.15, 0.0]),
            )
            + self._init_obj_pos
        )

        # Target is a different XY position ON THE TABLE
        target_xy = jax.random.uniform(
            rng_target,
            (2,),
            minval=jp.array([-0.2, -0.2]),
            maxval=jp.array([0.2, 0.2]),
        ) + self._init_obj_pos[:2]
        target_pos = jp.concatenate([target_xy, self._init_obj_pos[2:3]])

        init_q = (
            jp.array(self._init_q)
            .at[self._obj_qposadr : self._obj_qposadr + 3]
            .set(box_pos)
        )
        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        )

        metrics = {
            "out_of_bounds": jp.array(0.0),
            "success": jp.array(0.0),
            **{k: jp.array(0.0) for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "lift_gate": jp.array(0.0),  # smooth sticky: ramps 0→1 as max height → LIFT_HEIGHT
        }
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        box_pos = data.xpos[self._obj_body]
        target_pos = state.info["target_pos"]
        finger_dist = data.qpos[self._robot_qposadr[-1]] + data.qpos[self._robot_qposadr[-2]]

        # Smooth sticky lift gate — ramps 0→1 as max height approaches LIFT_HEIGHT.
        # Sticky: once it reaches a level, it never drops back (survives drops).
        # At 5cm: gate=0.5, at 10cm: gate=1.0. Gradual unlock, no hard wall.
        box_height = box_pos[2] - self._init_obj_pos[2]
        current_lift_gate = jp.clip(box_height / LIFT_HEIGHT, 0.0, 1.0)
        lift_gate = jp.maximum(state.info["lift_gate"], current_lift_gate)
        info = {**state.info, "lift_gate": lift_gate}

        raw_rewards = self._get_reward(data, info)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in raw_rewards.items()
        }
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0) | (box_pos[2] < 0.0)
        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        # Check success — requires lift_gate >= 0.95 (cube reached ~95% of LIFT_HEIGHT)
        xy_dist = jp.linalg.norm(box_pos[:2] - target_pos[:2])
        box_on_table = box_pos[2] < 0.06
        gripper_is_open = finger_dist > 0.06
        success = (
            (xy_dist < 0.03) & box_on_table & gripper_is_open & (lift_gate > 0.95)
        ).astype(float)

        state.metrics.update(
            **raw_rewards,
            out_of_bounds=out_of_bounds.astype(float),
            success=success,
        )

        obs = self._get_obs(data, info)
        return State(data, obs, reward, done, state.metrics, info)

    def _get_reward(self, data: mjx.Data, info: dict) -> dict:
        target_pos = info["target_pos"]
        lift_gate = info["lift_gate"]
        box_pos = data.xpos[self._obj_body]
        gripper_pos = data.site_xpos[self._gripper_site]
        finger_dist = data.qpos[self._robot_qposadr[-1]] + data.qpos[self._robot_qposadr[-2]]

        gripper_box_dist = jp.linalg.norm(box_pos - gripper_pos)
        box_target_xy_dist = jp.linalg.norm(box_pos[:2] - target_pos[:2])
        box_height = box_pos[2] - self._init_obj_pos[2]

        # Current-state detection (NOT sticky — recovers from drops)
        currently_grasping = (
            (gripper_box_dist < 0.04) & (finger_dist < 0.05)
        ).astype(float)

        # ── 1. REACH: Gripper approaches box (ALWAYS active) ──
        reach = 1 - jp.tanh(5.0 * gripper_box_dist)

        # ── 2. GRASP: Close fingers when near box (ALWAYS active) ──
        near_box = (gripper_box_dist < 0.03).astype(float)
        grasp = near_box * (1 - jp.tanh(15.0 * finger_dist))

        # ── 3. LIFT: Reward height while grasping (always active) ──
        # No (1-lift_gate) gating — lift always rewards being high while grasping.
        # place >> lift ensures the robot WANTS to eventually lower to the table.
        lift_progress = jp.clip(box_height / LIFT_HEIGHT, 0.0, 1.0)
        lift = lift_progress * currently_grasping

        # ── 4. TRANSPORT: Move box toward target XY (scaled by lift_gate) ──
        # Smoothly unlocks as the robot lifts higher. At 5cm: 50% reward. At 10cm: full.
        transport = (1 - jp.tanh(3.0 * box_target_xy_dist)) * currently_grasping * lift_gate

        # ── 5. PLACE: Box at target XY AND lowered to table (scaled by lift_gate) ──
        # Gated on XY proximity — no lowering reward until box is over the target.
        close_to_target_xy = jp.clip(1.0 - box_target_xy_dist / 0.1, 0.0, 1.0)
        height_above_target = jp.clip(box_pos[2] - target_pos[2], 0.0, 0.2)
        lowering = 1 - jp.tanh(8.0 * height_above_target)
        place = close_to_target_xy * lowering * lift_gate

        # ── 6. RELEASE: Open gripper when box is at target on table (scaled by lift_gate) ──
        at_target_on_table = (
            (box_target_xy_dist < 0.05) & (box_pos[2] < 0.06)
        ).astype(float)
        release = at_target_on_table * jp.clip(finger_dist / 0.08, 0.0, 1.0) * lift_gate

        # ── 7. SUCCESS BONUS (requires near-full lift) ──
        success_bonus = (
            (box_target_xy_dist < 0.03)
            & (box_pos[2] < 0.06)
            & (finger_dist > 0.06)
            & (lift_gate > 0.95)
        ).astype(float)

        # ── Regularization ──
        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Floor collision penalty
        hand_floor_collision = [
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._floor_hand_found_sensor
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision = (1 - floor_collision).astype(float)

        return {
            "reach": reach,
            "grasp": grasp,
            "lift": lift,
            "transport": transport,
            "place": place,
            "release": release,
            "success_bonus": success_bonus,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        gripper_pos = data.site_xpos[self._gripper_site]
        box_pos = data.xpos[self._obj_body]
        target_pos = info["target_pos"]
        finger_dist = data.qpos[self._robot_qposadr[-1]] + data.qpos[self._robot_qposadr[-2]]
        gripper_box_dist = jp.linalg.norm(box_pos - gripper_pos)

        # Current-state indicators
        currently_grasping = (
            (gripper_box_dist < 0.04) & (finger_dist < 0.05)
        ).astype(float)

        return jp.concatenate([
            data.qpos,
            data.qvel,
            gripper_pos,
            data.site_xmat[self._gripper_site].ravel()[3:],
            box_pos,
            box_pos - gripper_pos,
            target_pos - box_pos,
            target_pos - gripper_pos,
            jp.array([finger_dist]),
            jp.array([currently_grasping]),
            jp.array([info["lift_gate"]]),
            data.ctrl - data.qpos[self._robot_qposadr[:-1]],
        ])


# ---- Training ----

def main():
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground._src import wrapper

    print("Creating PandaPlaceCube environment...")
    env = PandaPlaceCube()
    print(f"  observation_size: {env.observation_size}")
    print(f"  action_size: {env.action_size}")

    # Quick sanity check
    print("\nSanity check...")
    rng = jax.random.key(0)
    state = jax.jit(env.reset)(rng)
    print(f"  obs shape: {state.obs.shape}")
    state2 = jax.jit(env.step)(state, jp.zeros(env.action_size))
    print(f"  step reward: {state2.reward}")
    print("  OK")

    # PPO config
    num_timesteps = 100_000_000
    print(f"\nTraining for {num_timesteps:,} timesteps...")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(64, 64, 64, 64),
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
        metrics_history.append({"steps": num_steps, "reward": float(reward)})
        print(f"  [{elapsed:6.1f}s] step={num_steps:>12,}  eval_reward={reward:.3f}  success={success}")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=20,
        reward_scaling=0.5,
        episode_length=300,
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

    eval_env = PandaPlaceCube()

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if metrics_history:
        print(f"Final eval reward: {metrics_history[-1]['reward']:.3f}")
        print(f"Best eval reward:  {max(m['reward'] for m in metrics_history):.3f}")

    # ---- Render GIF of trained policy ----
    print("\n--- Rendering GIF of trained policy ---")
    from PIL import Image

    gif_env = PandaPlaceCube()
    inference_fn = make_inference_fn(params)
    jit_inference = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)
    state = gif_env.reset(reset_rng)
    trajectory = [state]

    @jax.jit
    def gif_step(state, action):
        return gif_env.step(state, action)

    num_gif_steps = 300
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
    final_success = float(trajectory[-1].metrics.get("success", 0.0))
    print(f"  Episode reward: {total_reward:.1f}, success: {final_success}")

    print("  Rendering frames...")
    frames = gif_env.render(trajectory, height=480, width=640)
    print(f"  Got {len(frames)} frames")

    gif_path = "panda_place_trained.gif"
    images = [Image.fromarray(f) for f in frames[::2]]
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
