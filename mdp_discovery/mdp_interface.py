"""MDP Interface loader, validator, and runtime representation."""

import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp


@dataclass
class MDPInterface:
    """A loaded MDP interface with validated functions.

    Attributes:
        get_observation: (State) -> jnp.ndarray of shape (obs_dim,)
        compute_reward: (State, action, State) -> scalar jnp.float32
        obs_dim: detected observation dimension (set after validate())
        source_code: the raw Python source
    """

    get_observation: Callable
    compute_reward: Callable
    obs_dim: Optional[int] = None
    source_code: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "MDPInterface":
        """Load an MDP interface from a Python file."""
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

        if not hasattr(module, "get_observation"):
            raise AttributeError("MDP interface missing 'get_observation' function")
        if not hasattr(module, "compute_reward"):
            raise AttributeError("MDP interface missing 'compute_reward' function")

        source_code = Path(path).read_text()

        return cls(
            get_observation=module.get_observation,
            compute_reward=module.compute_reward,
            source_code=source_code,
        )

    @classmethod
    def from_code(cls, code: str) -> "MDPInterface":
        """Load an MDP interface from a code string."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp.flush()
            interface = cls.from_file(tmp.name)
            interface.source_code = code
            return interface

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def detect_obs_dim(self, dummy_state) -> int:
        """Call get_observation on a state and return the output size."""
        obs = self.get_observation(dummy_state)
        obs = jnp.asarray(obs)
        if obs.ndim != 1:
            raise ValueError(
                f"get_observation must return a 1D array, got ndim={obs.ndim}"
            )
        self.obs_dim = int(obs.shape[0])
        return self.obs_dim

    def validate(self, dummy_state, max_obs_dim: int = 512) -> int:
        """Full validation: check shapes, types, and constraints.

        Returns the detected obs_dim.
        """
        # -- get_observation --
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

        # -- compute_reward --
        reward = self.compute_reward(dummy_state, jnp.int32(0), dummy_state)
        reward = jnp.asarray(reward)
        if reward.ndim > 1:
            raise ValueError(
                f"compute_reward must return a scalar, got shape {reward.shape}"
            )
        if reward.ndim == 1 and reward.shape[0] != 1:
            raise ValueError(
                f"compute_reward must return a scalar, got shape {reward.shape}"
            )

        self.obs_dim = int(obs.shape[0])
        return self.obs_dim
