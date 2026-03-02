"""MuJoCo environment adapter for Brax PPO training.

Supports PandaTracking and Go1PushRecovery tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from mdp_discovery.adapters.base import EnvAdapter

logger = logging.getLogger(__name__)

# Map env_id strings to task classes (lazy imports to avoid heavy deps at import time)
_TASK_REGISTRY = {
    "PandaPickAndTrack": "mdp_discovery.tasks.panda_pick_and_track.PandaPickAndTrack",
    "Go1PushRecovery": "mdp_discovery.tasks.go1_push_recovery.Go1PushRecovery",
}


def _import_class(dotpath: str):
    """Import a class from a dotted path string."""
    module_path, class_name = dotpath.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class MjxMDPInterfaceWrapper:
    """Wraps a MuJoCo env, intercepts state.obs and state.reward with evolved functions.

    This is the MuJoCo equivalent of XMinigrid's MDPInterfaceWrapper.
    The wrapper is transparent to Brax's training pipeline — it produces
    the same State dataclass with modified obs and reward fields.
    """

    def __init__(self, env, get_obs_fn=None, reward_fn=None, obs_dim=None):
        self._env = env
        self._get_obs_fn = get_obs_fn
        self._reward_fn = reward_fn
        self._obs_dim = obs_dim

    def reset(self, rng):
        state = self._env.reset(rng)
        if self._get_obs_fn is not None:
            new_obs = self._get_obs_fn(state)
            new_obs = jnp.asarray(new_obs)
            state = state.replace(obs={"state": new_obs})
        return state

    def step(self, state, action):
        prev_state = state
        next_state = self._env.step(state, action)
        if self._get_obs_fn is not None:
            new_obs = self._get_obs_fn(next_state)
            new_obs = jnp.asarray(new_obs)
            next_state = next_state.replace(obs={"state": new_obs})
        if self._reward_fn is not None:
            new_reward = self._reward_fn(prev_state, action, next_state)
            new_reward = jnp.asarray(new_reward).squeeze()
            next_state = next_state.replace(reward=new_reward)
        return next_state

    @property
    def observation_size(self):
        """Report observation size for Brax PPO network construction."""
        if self._get_obs_fn is not None and self._obs_dim is not None:
            return {"state": (self._obs_dim,)}
        return self._env.observation_size

    @property
    def action_size(self):
        return self._env.action_size

    def __getattr__(self, name):
        """Delegate everything else to the wrapped env."""
        return getattr(self._env, name)


class MujocoAdapter(EnvAdapter):
    """Adapter for MuJoCo Playground environments using Brax PPO."""

    def __init__(self, config):
        self.config = config
        self._env_id = config.environment.env_id
        if self._env_id not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown MuJoCo env_id: {self._env_id!r}. "
                f"Available: {list(_TASK_REGISTRY.keys())}"
            )
        self._task_cls = _import_class(_TASK_REGISTRY[self._env_id])
        self._env = self._task_cls()

    def make_task_env(self):
        """Create a fresh task environment instance (public API for train_brax)."""
        return self._task_cls()

    def _make_wrapped_env(self, get_obs_fn, reward_fn, obs_dim=None):
        """Create a fresh env instance wrapped with MDP interface."""
        env = self.make_task_env()
        return MjxMDPInterfaceWrapper(env, get_obs_fn, reward_fn, obs_dim=obs_dim)

    def make_env(self, get_obs_fn, reward_fn) -> Tuple[Any, Any]:
        wrapped = self._make_wrapped_env(get_obs_fn, reward_fn)
        # env_params is None for MuJoCo (Brax manages its own params)
        return wrapped, None

    def make_eval_env(self, get_obs_fn, reward_fn, max_steps: int) -> Tuple[Any, Any]:
        # max_steps is handled by Brax's EpisodeWrapper, not env_params
        wrapped = self._make_wrapped_env(get_obs_fn, reward_fn)
        return wrapped, None

    def get_dummy_state(self) -> Any:
        """Get a real environment state for dry-run validation."""
        rng = jax.random.key(0)
        state = jax.jit(self._env.reset)(rng)
        return state

    def get_dummy_action(self) -> jax.Array:
        """Return a zero continuous action vector."""
        return jnp.zeros(self._env.action_size)

    def is_continuous_action(self) -> bool:
        return True

    def compute_success(self, rollout_stats: Any, env_params: Any) -> jax.Array:
        """Compute per-episode success.

        For MuJoCo tasks using Brax PPO, success is reported via the progress
        callback (eval/episode_success) rather than this method. This is kept
        for API compatibility with the base class.
        """
        if hasattr(rollout_stats, "metrics") and isinstance(rollout_stats.metrics, dict):
            success = rollout_stats.metrics.get("success")
            if success is not None:
                return success
        # Fallback
        return (rollout_stats.reward > 0.5).astype(jnp.float32)

    def get_default_obs_fn(self) -> Callable:
        """Return the env's built-in observation as a function of state.

        Both Go1 (dict obs {"state": array}) and Panda (flat array) are
        handled. The returned function extracts the flat obs vector.
        """
        # Detect obs format once at construction time, not at trace time
        env = self._env
        rng = jax.random.key(0)
        dummy_state = jax.eval_shape(env.reset, rng)
        obs_is_dict = isinstance(dummy_state.obs, dict)

        if obs_is_dict:
            def default_obs(state):
                return state.obs["state"]
        else:
            def default_obs(state):
                return state.obs

        return default_obs

    def get_default_reward_fn(self) -> Optional[Callable]:
        """Return None — use the env's built-in reward."""
        return None

    def num_actions(self, env_params: Any) -> int:
        return self._env.action_size

    # ------------------------------------------------------------------
    # Training dispatch
    # ------------------------------------------------------------------

    def run_training(self, config, interface, obs_dim, total_timesteps=None) -> Dict[str, Any]:
        """Run Brax PPO training (single seed)."""
        from mdp_discovery.train_brax import run_training
        return run_training(config, self, interface, obs_dim, total_timesteps)

    def run_training_multi_seed(self, config, interface, obs_dim, total_timesteps=None, num_seeds=None) -> Dict[str, Any]:
        """Run Brax PPO training across multiple seeds without recompiling."""
        from mdp_discovery.train_brax import run_training_multi_seed
        return run_training_multi_seed(config, self, interface, obs_dim, total_timesteps, num_seeds)
