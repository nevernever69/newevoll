"""MDP Interface loader, validator, and runtime representation."""

import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp


@dataclass
class MDPInterface:
    """A loaded MDP interface with validated functions.

    Attributes:
        get_observation: (State) -> jnp.ndarray of shape (obs_dim,), or None if not required.
        compute_reward: (State, action, State) -> scalar jnp.float32, or None if not required.
        obs_dim: detected observation dimension (set after validate())
        source_code: the raw Python source
    """

    get_observation: Optional[Callable] = None
    compute_reward: Optional[Callable] = None
    obs_dim: Optional[int] = None
    source_code: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str,
        required_functions: Optional[List[str]] = None,
    ) -> "MDPInterface":
        """Load an MDP interface from a Python file.

        Args:
            path: Path to the Python file.
            required_functions: Functions that must exist. Defaults to both.
                Missing non-required functions are set to None.
        """
        if required_functions is None:
            required_functions = ["get_observation", "compute_reward"]

        path = str(path)
        module_name = f"_mdp_interface_{id(path)}"

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        for fn_name in required_functions:
            if not hasattr(module, fn_name):
                raise AttributeError(f"MDP interface missing '{fn_name}' function")

        source_code = Path(path).read_text()

        return cls(
            get_observation=getattr(module, "get_observation", None),
            compute_reward=getattr(module, "compute_reward", None),
            source_code=source_code,
        )

    @classmethod
    def from_code(
        cls,
        code: str,
        required_functions: Optional[List[str]] = None,
    ) -> "MDPInterface":
        """Load an MDP interface from a code string."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp.flush()
            interface = cls.from_file(tmp.name, required_functions=required_functions)
            interface.source_code = code
            return interface

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def detect_obs_dim(self, dummy_state) -> int:
        """Call get_observation on a state and return the output size."""
        if self.get_observation is None:
            raise ValueError("Cannot detect obs_dim: get_observation is None")
        obs = self.get_observation(dummy_state)
        obs = jnp.asarray(obs)
        if obs.ndim != 1:
            raise ValueError(
                f"get_observation must return a 1D array, got ndim={obs.ndim}"
            )
        self.obs_dim = int(obs.shape[0])
        return self.obs_dim

    def validate(self, dummy_state, max_obs_dim: int = 512, dummy_action=None) -> int:
        """Full validation: check shapes, types, and constraints.

        Only validates functions that are not None.
        Returns the detected obs_dim (0 if get_observation is None).

        Args:
            dummy_state: A real environment state for dry-run validation.
            max_obs_dim: Maximum allowed observation dimension.
            dummy_action: Action to use for compute_reward validation.
                Defaults to jnp.int32(0) for discrete envs.
        """
        if dummy_action is None:
            dummy_action = jnp.int32(0)

        obs_dim = 0

        # -- get_observation --
        if self.get_observation is not None:
            obs = self.get_observation(dummy_state)
            obs = jnp.asarray(obs)
            if obs.ndim != 1:
                raise ValueError(
                    f"get_observation must return a 1D array, got shape {obs.shape}"
                )
            if obs.shape[0] > max_obs_dim:
                raise ValueError(
                    f"Observation dim {obs.shape[0]} exceeds max_obs_dim={max_obs_dim}"
                )
            if obs.shape[0] == 0:
                raise ValueError("get_observation returned an empty array")
            obs_dim = int(obs.shape[0])

        # -- compute_reward --
        if self.compute_reward is not None:
            reward = self.compute_reward(dummy_state, dummy_action, dummy_state)
            reward = jnp.asarray(reward)
            if reward.ndim > 1:
                raise ValueError(
                    f"compute_reward must return a scalar, got shape {reward.shape}"
                )
            if reward.ndim == 1 and reward.shape[0] != 1:
                raise ValueError(
                    f"compute_reward must return a scalar, got shape {reward.shape}"
                )

        self.obs_dim = obs_dim
        return self.obs_dim
