#!/usr/bin/env python3
"""
Optional W&B integration for MDP Discovery.
Add this to log metrics to Weights & Biases.

Usage:
    pip install wandb
    wandb login

Then modify controller.py to call these functions.
"""

import os
from typing import Optional, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not installed. Run: pip install wandb")


class WandBLogger:
    """Optional W&B logger for MDP Discovery runs."""

    def __init__(
        self,
        project: str = "mdp-discovery",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and WANDB_AVAILABLE

        if self.enabled:
            wandb.init(
                project=project,
                name=name,
                config=config,
                tags=["ablation", "no-evolution"] if "noevo" in str(name) else [],
            )

    def log_candidate(
        self,
        iteration: int,
        candidate_num: int,
        metrics: Dict[str, Any],
        obs_dim: int,
        stage: str,
    ):
        """Log a single candidate evaluation."""
        if not self.enabled:
            return

        wandb.log({
            "iteration": iteration,
            "candidate": candidate_num,
            "success_rate": metrics.get("success_rate", 0.0),
            "episode_length": metrics.get("final_length", 0.0),
            "obs_dim": obs_dim,
            "stage": stage,
            "training_time": metrics.get("training_time", 0.0),
        })

    def log_best(self, iteration: int, success_rate: float, obs_dim: int):
        """Log new best candidate."""
        if not self.enabled:
            return

        wandb.log({
            "iteration": iteration,
            "best_success_rate": success_rate,
            "best_obs_dim": obs_dim,
        })

    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary."""
        if not self.enabled:
            return

        wandb.summary.update(summary)

    def finish(self):
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()


# Example integration in controller.py:
"""
# In EvolutionController.__init__:
self.wandb_logger = WandBLogger(
    project="mdp-discovery-ablation",
    name=f"{config.environment.env_id}_noevo",
    config=config.to_dict(),
    enabled=os.getenv("WANDB_ENABLED", "0") == "1",
)

# In _log_iteration:
if program:
    self.wandb_logger.log_candidate(
        iteration=self.iteration,
        candidate_num=candidate_num,
        metrics=result.metrics,
        obs_dim=program.obs_dim,
        stage=result.stage.value,
    )

# When new best found:
self.wandb_logger.log_best(
    iteration=self.iteration,
    success_rate=best.fitness,
    obs_dim=best.obs_dim,
)

# In run() finally block:
self.wandb_logger.finish()
"""
