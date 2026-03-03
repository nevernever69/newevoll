"""Discovery-ready Go1PushRecovery environment.

Thin subclass that adds pre-computed sensor readings to state.info so the LLM
can access them without needing env helper methods like self.get_gyro(data).
"""

import sys
import os

# Ensure both repo root (for test_go1_pushrecovery) and mujoco_playground are importable
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_playground_path = os.path.join(_repo_root, "mujoco_playground")
if _playground_path not in sys.path:
    sys.path.insert(0, _playground_path)

import jax
import jax.numpy as jp
from mujoco_playground._src.mjx_env import State

from test_go1_pushrecovery import Go1PushRecovery as _BaseGo1PushRecovery


class Go1PushRecovery(_BaseGo1PushRecovery):
    """Go1PushRecovery with enriched state.info for LLM-evolved interfaces.

    Adds to state.info:
        gyro: (3,) angular velocity from IMU
        gravity: (3,) gravity vector in body frame
        local_linvel: (3,) local linear velocity
        upvector: (3,) body up-vector (upvector[-1] < 0.3 = fallen)
        pos_xy: (2,) XY offset from origin
        height: () COM height
        heading: () current yaw angle
    """

    def _enrich_info(self, data, info):
        """Add pre-computed sensor readings to info dict."""
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        local_linvel = self.get_local_linvel(data)
        upvector = self.get_upvector(data)
        pos_xy = data.qpos[0:2] - self._origin_xy
        height = data.qpos[2]
        heading = self._quat_to_yaw(data.qpos[3:7])

        return {
            **info,
            "gyro": gyro,
            "gravity": gravity,
            "local_linvel": local_linvel,
            "upvector": upvector,
            "pos_xy": pos_xy,
            "height": height,
            "heading": heading,
        }

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        info = self._enrich_info(state.data, state.info)
        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)
        info = self._enrich_info(state.data, state.info)
        return state.replace(info=info)
