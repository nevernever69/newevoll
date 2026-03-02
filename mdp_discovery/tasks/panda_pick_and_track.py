"""Discovery-ready PandaTracking environment.

Thin subclass that adds pre-computed values to state.info so the LLM
can access them without needing env helper methods.
"""

import sys
import os

# Ensure both repo root (for test_panda_tracking) and mujoco_playground are importable
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_playground_path = os.path.join(_repo_root, "mujoco_playground")
if _playground_path not in sys.path:
    sys.path.insert(0, _playground_path)

import jax
import jax.numpy as jp
from mujoco_playground._src.mjx_env import State

from test_panda_tracking import PandaTracking as _BasePandaTracking


class PandaPickAndTrack(_BasePandaTracking):
    """PandaTracking with enriched state.info for LLM-evolved interfaces.

    Adds to state.info:
        target_pos: (3,) current Lissajous target position
        target_vel: (3,) current target velocity
        gripper_pos: (3,) end-effector position
        gripper_target_dist: () distance between gripper and target
    """

    def _enrich_info(self, data, info):
        """Add pre-computed values to info dict."""
        t = info["step_count"] * self._config.ctrl_dt
        traj_params = info["traj_params"]
        target_pos = self._get_target_pos(t, traj_params)
        target_vel = self._get_target_vel(t, traj_params)
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_target_dist = jp.linalg.norm(target_pos - gripper_pos)

        return {
            **info,
            "target_pos": target_pos,
            "target_vel": target_vel,
            "gripper_pos": gripper_pos,
            "gripper_target_dist": gripper_target_dist,
        }

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        info = self._enrich_info(state.data, state.info)
        return state.replace(info=info)

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)
        info = self._enrich_info(state.data, state.info)
        return state.replace(info=info)
