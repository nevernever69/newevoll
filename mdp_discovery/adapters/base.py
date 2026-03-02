"""Abstract base class defining the environment adapter interface."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


class EnvAdapter(ABC):
    """Protocol for environment adapters.

    Encapsulates all environment-specific logic: creation, wrapping,
    dummy state generation, success computation, and default functions.
    """

    @abstractmethod
    def make_env(
        self,
        get_obs_fn: Optional[Callable],
        reward_fn: Optional[Callable],
    ) -> Tuple[Any, Any]:
        """Create a wrapped environment ready for training.

        Args:
            get_obs_fn: Custom observation function, or None to use env default.
            reward_fn: Custom reward function, or None to use env default.

        Returns:
            (env, env_params) tuple.
        """

    @abstractmethod
    def make_eval_env(
        self,
        get_obs_fn: Optional[Callable],
        reward_fn: Optional[Callable],
        max_steps: int,
    ) -> Tuple[Any, Any]:
        """Create a wrapped environment for evaluation with custom max_steps.

        Args:
            get_obs_fn: Custom observation function, or None to use env default.
            reward_fn: Custom reward function, or None to use env default.
            max_steps: Maximum steps per episode for evaluation.

        Returns:
            (env, env_params) tuple with max_steps overridden.
        """

    @abstractmethod
    def get_dummy_state(self) -> Any:
        """Get a real environment state for dry-run validation.

        Returns:
            A state object from resetting the environment.
        """

    @abstractmethod
    def compute_success(self, rollout_stats: Any, env_params: Any) -> jax.Array:
        """Compute per-episode success from rollout statistics.

        Args:
            rollout_stats: Statistics from evaluation rollouts.
            env_params: Environment parameters (for max_steps etc.).

        Returns:
            Float array of per-episode success (1.0 = success, 0.0 = failure).
        """

    @abstractmethod
    def get_default_obs_fn(self) -> Callable:
        """Return the default (no-design-effort) observation function.

        This is used as the baseline observation when only reward is evolved.
        """

    @abstractmethod
    def get_default_reward_fn(self) -> Optional[Callable]:
        """Return the default reward function, or None to use env built-in.

        Returns None when the environment's built-in reward should be used
        (e.g., for obs_only ablation mode).
        """

    @abstractmethod
    def num_actions(self, env_params: Any) -> int:
        """Return the number of discrete actions for this environment."""

    # ------------------------------------------------------------------
    # Optional overrides (backward-compatible defaults for discrete envs)
    # ------------------------------------------------------------------

    def run_training(self, config, interface, obs_dim, total_timesteps=None) -> Dict[str, Any]:
        """Run RL training. Discrete adapters use train.py; MuJoCo uses train_brax.py."""
        from mdp_discovery.train import run_training
        return run_training(config, self, interface, obs_dim, total_timesteps)

    def run_training_multi_seed(self, config, interface, obs_dim, total_timesteps=None, num_seeds=None) -> Dict[str, Any]:
        """Run RL training across multiple seeds.

        Default: loops over run_training() with different seeds.
        MuJoCo adapter overrides this to share compiled train_fn across seeds.
        """
        import dataclasses
        tc = config.training
        n = num_seeds or tc.num_seeds
        if n <= 1:
            return self.run_training(config, interface, obs_dim, total_timesteps)

        all_success, all_return, all_length, all_curves = [], [], [], []
        total_time = 0.0
        for i in range(n):
            seed = tc.seed + i
            mod_config = dataclasses.replace(
                config, training=dataclasses.replace(tc, seed=seed),
            )
            m = self.run_training(mod_config, interface, obs_dim, total_timesteps)
            all_success.append(m["success_rate"])
            all_return.append(m["final_return"])
            all_length.append(m["final_length"])
            total_time += m.get("training_time", 0.0)
            if m.get("success_curve"):
                all_curves.append(m["success_curve"])

        avg_curve = []
        if all_curves:
            min_len = min(len(c) for c in all_curves)
            avg_curve = [sum(c[j] for c in all_curves) / n for j in range(min_len)]

        return {
            "final_return": sum(all_return) / n,
            "final_length": sum(all_length) / n,
            "success_rate": sum(all_success) / n,
            "success_curve": avg_curve,
            "learning_curve": [],
            "training_time": total_time,
        }

    def get_dummy_action(self) -> jax.Array:
        """Return a dummy action for crash filter validation."""
        return jnp.int32(0)

    def is_continuous_action(self) -> bool:
        """Whether the action space is continuous."""
        return False
