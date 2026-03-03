"""W&B integration for MDP Discovery.

Optional Weights & Biases logging for evolution runs.
Tracks candidates, success rates, best programs, and more.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandBLogger:
    """Optional W&B logger for MDP Discovery runs."""

    def __init__(
        self,
        project: str = "mdp-discovery",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        enabled: bool = True,
    ):
        """Initialize W&B logger.

        Args:
            project: W&B project name
            name: Run name (auto-generated if None)
            config: Configuration dict to log
            tags: Tags for the run
            enabled: Enable/disable logging (controlled by WANDB_ENABLED env var)
        """
        # Check if enabled via environment variable
        wandb_env = os.getenv("WANDB_ENABLED", "0").lower()
        self.enabled = enabled and wandb_env in ("1", "true", "yes") and WANDB_AVAILABLE

        if not WANDB_AVAILABLE and wandb_env in ("1", "true", "yes"):
            logger.warning(
                "W&B logging requested but wandb not installed. "
                "Install with: pip install wandb"
            )

        self.run = None

        if self.enabled:
            try:
                self.run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    tags=tags or [],
                    reinit=True,
                )
                logger.info("W&B logging enabled: %s/%s", project, name)
            except Exception as e:
                logger.warning("Failed to initialize W&B: %s", e)
                self.enabled = False

    def log_candidate(
        self,
        iteration: int,
        candidate_num: int,
        total_candidates: int,
        metrics: Dict[str, Any],
        obs_dim: int,
        stage: str,
        is_best: bool = False,
    ):
        """Log a single candidate evaluation.

        Args:
            iteration: Current iteration number
            candidate_num: Candidate number within batch
            total_candidates: Total candidates in batch
            metrics: Training metrics dict
            obs_dim: Observation dimension
            stage: Evaluation stage (crash_filter, short_train, full_train)
            is_best: Whether this is the new best candidate
        """
        if not self.enabled or not self.run:
            return

        try:
            log_dict = {
                "iteration": iteration,
                "candidate": candidate_num,
                "candidate_progress": candidate_num / total_candidates,
                "success_rate": metrics.get("success_rate", 0.0),
                "episode_length": metrics.get("final_length", 0.0),
                "obs_dim": obs_dim,
                "stage": stage,
                "training_time": metrics.get("training_time", 0.0),
            }

            if is_best:
                log_dict["new_best"] = 1

            wandb.log(log_dict)
        except Exception as e:
            logger.debug("W&B log failed: %s", e)

    def log_best(
        self,
        iteration: int,
        success_rate: float,
        obs_dim: int,
        episode_length: float,
    ):
        """Log new best candidate.

        Args:
            iteration: Current iteration
            success_rate: Success rate of best candidate
            obs_dim: Observation dimension
            episode_length: Average episode length
        """
        if not self.enabled or not self.run:
            return

        try:
            wandb.log(
                {
                    "iteration": iteration,
                    "best/success_rate": success_rate,
                    "best/obs_dim": obs_dim,
                    "best/episode_length": episode_length,
                }
            )
        except Exception as e:
            logger.debug("W&B log failed: %s", e)

    def log_iteration_summary(
        self,
        iteration: int,
        candidates_generated: int,
        candidates_passed: int,
        candidates_crashed: int,
        best_success_rate: float,
        avg_eval_time: float,
        total_llm_tokens: int,
    ):
        """Log iteration-level summary statistics.

        Args:
            iteration: Current iteration
            candidates_generated: Number of candidates generated
            candidates_passed: Number that passed crash filter
            candidates_crashed: Number that crashed
            best_success_rate: Best success rate so far
            avg_eval_time: Average evaluation time per candidate
            total_llm_tokens: Cumulative LLM tokens used
        """
        if not self.enabled or not self.run:
            return

        try:
            wandb.log(
                {
                    "iteration": iteration,
                    "summary/candidates_generated": candidates_generated,
                    "summary/candidates_passed": candidates_passed,
                    "summary/candidates_crashed": candidates_crashed,
                    "summary/pass_rate": (
                        candidates_passed / max(candidates_generated, 1)
                    ),
                    "summary/best_success_rate": best_success_rate,
                    "summary/avg_eval_time": avg_eval_time,
                    "summary/total_llm_tokens": total_llm_tokens,
                }
            )
        except Exception as e:
            logger.debug("W&B log failed: %s", e)

    def log_final_summary(self, summary: Dict[str, Any]):
        """Log final run summary.

        Args:
            summary: Dict with final statistics
        """
        if not self.enabled or not self.run:
            return

        try:
            wandb.summary.update(summary)
        except Exception as e:
            logger.debug("W&B summary update failed: %s", e)

    def log_artifact(self, artifact_path: str, artifact_type: str, name: str):
        """Log a file artifact (e.g., best_interface.py).

        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact ("code", "model", etc.)
            name: Artifact name
        """
        if not self.enabled or not self.run:
            return

        try:
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            logger.debug("W&B artifact log failed: %s", e)

    def finish(self):
        """Finish W&B run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.debug("W&B finish failed: %s", e)
