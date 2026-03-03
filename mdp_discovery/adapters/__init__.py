"""Environment adapter factory and exports."""

from mdp_discovery.adapters.base import EnvAdapter
from mdp_discovery.adapters.xminigrid_adapter import XMinigridAdapter


def get_adapter(adapter_type: str, config) -> EnvAdapter:
    """Create an environment adapter by type name.

    Args:
        adapter_type: Adapter identifier (e.g., "xminigrid", "mujoco").
        config: Full system Config object.

    Returns:
        An EnvAdapter instance.
    """
    if adapter_type == "xminigrid":
        return XMinigridAdapter(config)
    elif adapter_type == "mujoco":
        from mdp_discovery.adapters.mujoco_adapter import MujocoAdapter
        return MujocoAdapter(config)
    raise ValueError(f"Unknown adapter type: {adapter_type!r}")
