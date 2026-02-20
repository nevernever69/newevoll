"""XMinigrid environment adapter.

All xminigrid-specific code lives in this single file.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper, Wrapper

from mdp_discovery.adapters.base import EnvAdapter


# ---------------------------------------------------------------------------
# MDPInterfaceWrapper (moved from wrapper.py)
# ---------------------------------------------------------------------------


class MDPInterfaceWrapper(Wrapper):
    """Replaces the default observation and/or reward with MDP interface outputs.

    Wrapper stacking order (important!):
        env = MDPInterfaceWrapper(env, get_obs_fn, compute_reward_fn)  # first
        env = GymAutoResetWrapper(env)                                  # second

    This ensures compute_reward sees the true terminal state on episode
    boundaries, before GymAutoResetWrapper replaces it with the reset state.

    If get_observation_fn is None, the original timestep.observation is kept.
    If compute_reward_fn is None, the original timestep.reward is kept.
    """

    def __init__(
        self,
        env,
        get_observation_fn: Optional[Callable] = None,
        compute_reward_fn: Optional[Callable] = None,
    ):
        super().__init__(env)
        self._get_observation = get_observation_fn
        self._compute_reward = compute_reward_fn

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        if self._get_observation is not None:
            new_obs = self._get_observation(timestep.state)
            return timestep.replace(observation=new_obs)
        return timestep

    def step(self, params, timestep, action):
        prev_state = timestep.state
        new_timestep = self._env.step(params, timestep, action)
        replacements = {}
        if self._get_observation is not None:
            replacements["observation"] = self._get_observation(new_timestep.state)
        if self._compute_reward is not None:
            replacements["reward"] = self._compute_reward(
                prev_state, action, new_timestep.state
            )
        if replacements:
            return new_timestep.replace(**replacements)
        return new_timestep


# ---------------------------------------------------------------------------
# XMinigrid Adapter
# ---------------------------------------------------------------------------


class XMinigridAdapter(EnvAdapter):
    """Adapter for xland-minigrid environments."""

    def __init__(self, config):
        self.config = config
        self._env_cfg = config.environment

    @staticmethod
    def _load_ruleset_file(path: str):
        """Load a RuleSet from a Python file that defines build_ruleset()."""
        spec = importlib.util.spec_from_file_location("_custom_ruleset", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "build_ruleset"):
            raise ValueError(f"Ruleset file {path} must define a build_ruleset() function")
        return mod.build_ruleset()

    def _make_base_env(self) -> Tuple[Any, Any]:
        """Create the base (unwrapped) environment and params."""
        env, env_params = xminigrid.make(self._env_cfg.env_id)

        if self._env_cfg.ruleset_file is not None:
            ruleset = self._load_ruleset_file(self._env_cfg.ruleset_file)
            env_params = env_params.replace(ruleset=ruleset)
        elif (
            self._env_cfg.benchmark_id is not None
            and self._env_cfg.ruleset_id is not None
        ):
            benchmark = xminigrid.load_benchmark(self._env_cfg.benchmark_id)
            env_params = env_params.replace(
                ruleset=benchmark.get_ruleset(self._env_cfg.ruleset_id)
            )

        return env, env_params

    def make_env(
        self,
        get_obs_fn: Optional[Callable],
        reward_fn: Optional[Callable],
    ) -> Tuple[Any, Any]:
        env, env_params = self._make_base_env()
        env = MDPInterfaceWrapper(env, get_obs_fn, reward_fn)
        env = GymAutoResetWrapper(env)
        return env, env_params

    def make_eval_env(
        self,
        get_obs_fn: Optional[Callable],
        reward_fn: Optional[Callable],
        max_steps: int,
    ) -> Tuple[Any, Any]:
        env, env_params = self._make_base_env()
        env_params = env_params.replace(max_steps=max_steps)
        env = MDPInterfaceWrapper(env, get_obs_fn, reward_fn)
        env = GymAutoResetWrapper(env)
        return env, env_params

    def get_dummy_state(self) -> Any:
        env, env_params = self._make_base_env()
        key = jax.random.key(0)
        timestep = env.reset(env_params, key)
        return timestep.state

    def compute_success(self, rollout_stats: Any, env_params: Any) -> jax.Array:
        return (rollout_stats.length < env_params.max_steps).astype(jnp.float32)

    def get_default_obs_fn(self) -> Callable:
        """Return a baseline observation: flattened grid + position + direction + pocket + step."""
        # Capture grid size from a dummy env for normalization
        env, env_params = self._make_base_env()
        key = jax.random.key(0)
        timestep = env.reset(env_params, key)
        grid_shape = timestep.state.grid.shape  # (H, W, 2)
        grid_size = max(grid_shape[0], grid_shape[1])

        def default_obs(state):
            flat_grid = state.grid.reshape(-1).astype(jnp.float32) / 12.0
            pos = state.agent.position.astype(jnp.float32) / (grid_size - 1)
            direction = jax.nn.one_hot(state.agent.direction, 4)
            pocket = state.agent.pocket.astype(jnp.float32) / 12.0
            step = jnp.array([state.step_num.astype(jnp.float32) / 100.0])
            return jnp.concatenate([flat_grid, pos, direction, pocket, step])

        return default_obs

    def get_default_reward_fn(self) -> Optional[Callable]:
        """Return None — signals wrapper to use the env's built-in reward."""
        return None

    def num_actions(self, env_params: Any) -> int:
        env, _ = self._make_base_env()
        return env.num_actions(env_params)
