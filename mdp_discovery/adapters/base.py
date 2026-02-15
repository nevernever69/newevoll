"""Abstract base class defining the environment adapter interface."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

import jax


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
