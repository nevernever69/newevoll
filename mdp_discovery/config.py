"""
Configuration system for MDP Interface Discovery.

Hierarchical dataclass-based config with YAML loading via dacite,
adapted from OpenEvolve's configuration pattern.
"""

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dacite
import yaml


_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$")


def _resolve_env_var(value: Optional[str]) -> Optional[str]:
    """Resolve ${VAR} environment variable reference in a string value.

    Pattern must match the entire string (e.g., "${OPENAI_API_KEY}"),
    not embedded within other text.
    """
    if value is None:
        return None
    match = _ENV_VAR_PATTERN.match(value)
    if not match:
        return value
    var_name = match.group(1)
    env_value = os.environ.get(var_name)
    if env_value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return env_value


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentConfig:
    """Configuration for the environment."""

    env_id: str = "MiniGrid-Empty-6x6"
    benchmark_id: Optional[str] = None
    ruleset_id: Optional[int] = None
    view_size: int = 7
    max_steps: Optional[int] = None
    adapter_type: str = "xminigrid"
    context_file: str = "contexts/xminigrid_context.md"


@dataclass
class TrainingConfig:
    """PPO training hyperparameters.

    Defaults match xland-minigrid's train_single_task.py.
    """

    # Parallelism
    num_envs: int = 8192
    num_steps: int = 16

    # PPO
    update_epochs: int = 1
    num_minibatches: int = 16
    total_timesteps: int = 1_000_000
    total_timesteps_full: int = 5_000_000
    lr: float = 0.001
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Evaluation
    eval_episodes: int = 80
    eval_episodes_for_fitness: int = 50
    eval_max_steps: int = 80
    seed: int = 42
    num_seeds: int = 3

    # Precision
    enable_bf16: bool = False

    # Network architecture
    hidden_dim: int = 256
    rnn_hidden_dim: int = 512
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    action_emb_dim: int = 16


@dataclass
class MDPInterfaceConfig:
    """Configuration for the MDP interface template."""

    max_obs_dim: int = 512
    include_action_transform: bool = False


@dataclass
class CrashFilterConfig:
    """Configuration for the pre-training crash filter."""

    stage0_timeout: int = 5
    stage1_timeout: int = 10
    stage2_timeout: int = 30
    retry_on_crash: bool = True
    max_crash_retries: int = 1


@dataclass
class EvaluatorConfig:
    """Cascade RL evaluation configuration."""

    timeout: int = 600
    max_retries: int = 2
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.05])
    parallel_evaluations: int = 1
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100 MB


@dataclass
class LLMConfig:
    """AWS Bedrock LLM configuration."""

    region_name: str = "us-east-1"
    model_name: str = "claude-4-sonnet"
    temperature: float = 0.7
    max_tokens: int = 4096
    retries: int = 3
    retry_delay: float = 1.0


@dataclass
class DatabaseConfig:
    """MAP-Elites + island evolution database configuration."""

    db_path: Optional[str] = None
    in_memory: bool = True
    population_size: int = 200
    archive_size: int = 50
    num_islands: int = 3
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    feature_dimensions: List[str] = field(
        default_factory=lambda: ["reward_complexity", "obs_dimensionality"]
    )
    feature_bins: int = 5
    migration_interval: int = 20
    migration_rate: float = 0.1
    random_seed: Optional[int] = 42


@dataclass
class PromptConfig:
    """Prompt construction configuration."""

    num_top_programs: int = 3
    num_diverse_programs: int = 2
    num_failure_examples: int = 2
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20 KB


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Root configuration for MDP Interface Discovery."""

    max_iterations: int = 500
    candidates_per_iteration: int = 1
    checkpoint_interval: int = 50
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: int = 42
    evolution_mode: str = "full"
    task_description: Optional[str] = None

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mdp_interface: MDPInterfaceConfig = field(default_factory=MDPInterfaceConfig)
    crash_filter: CrashFilterConfig = field(default_factory=CrashFilterConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        if config_dict is None:
            config_dict = {}
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Parse a dictionary into a Config instance."""
        return dacite.from_dict(
            data_class=cls,
            data=config_dict,
            config=dacite.Config(cast=[List]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
