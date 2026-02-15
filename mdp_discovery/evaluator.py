"""Cascade evaluator for MDP interface candidates.

Pipeline: crash filter → short training → threshold check → full training.
Returns a CandidateResult with metrics, crash info, and the evaluation stage reached.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from mdp_discovery.adapters.base import EnvAdapter
from mdp_discovery.config import Config
from mdp_discovery.crash_filter import CrashFilterResult, run_crash_filter
from mdp_discovery.mdp_interface import MDPInterface
from mdp_discovery.train import run_training

logger = logging.getLogger(__name__)


class EvalStage(Enum):
    """How far a candidate got through the evaluation pipeline."""

    CRASHED = "crashed"
    SHORT_TRAIN = "short_train"
    SHORT_TRAIN_REJECTED = "short_train_rejected"
    FULL_TRAIN = "full_train"


@dataclass
class CandidateResult:
    """Complete evaluation result for a single candidate.

    This is what gets stored in the database and fed back to the LLM.
    """

    code: str
    stage: EvalStage

    # Crash filter output (always set)
    crash_result: CrashFilterResult

    # Training metrics (set if training ran)
    metrics: Optional[Dict[str, Any]] = None

    # Obs dim detected by crash filter (set if passed crash filter)
    obs_dim: Optional[int] = None

    # Total wall-clock time for the full evaluation
    eval_time: float = 0.0

    @property
    def passed(self) -> bool:
        """Did the candidate pass crash filter and produce training metrics?"""
        return self.crash_result.passed and self.metrics is not None

    @property
    def fitness(self) -> float:
        """Success rate (fraction of eval episodes reaching the goal), or 0.0 if not trained."""
        if self.metrics is None:
            return 0.0
        return self.metrics.get("success_rate", 0.0)


class CascadeEvaluator:
    """Evaluates candidate MDP interfaces through a cascade of increasing cost.

    Stage 1: Crash filter (fast — syntax, import, dry-run)
    Stage 2: Short training (config.training.total_timesteps, default 1M steps)
    Stage 3: Full training (config.training.total_timesteps_full, default 5M steps)
             Only if short training passes the cascade threshold.

    When cascade_evaluation is disabled, goes directly to full training
    after passing the crash filter.
    """

    def __init__(self, config: Config, adapter: EnvAdapter):
        self.config = config
        self.adapter = adapter
        self.dummy_state = adapter.get_dummy_state()

    def _get_required_functions(self) -> List[str]:
        """Return required function names based on evolution mode."""
        mode = self.config.evolution_mode
        if mode in ("full", "random"):
            return ["get_observation", "compute_reward"]
        elif mode == "reward_only":
            return ["compute_reward"]
        elif mode == "obs_only":
            return ["get_observation"]
        elif mode == "default":
            return []
        else:
            return ["get_observation", "compute_reward"]

    def evaluate(self, code: str) -> CandidateResult:
        """Run the full cascade evaluation on a candidate.

        Args:
            code: Python source code defining get_observation and/or compute_reward.

        Returns:
            CandidateResult with metrics and stage info.
        """
        t_start = time.time()
        required_functions = self._get_required_functions()

        # --- Stage 1: Crash filter ---
        logger.info("Running crash filter...")
        crash_result = run_crash_filter(
            code, self.config, self.dummy_state,
            required_functions=required_functions,
        )

        if not crash_result.passed:
            logger.info(
                "Candidate crashed at stage %s: %s",
                crash_result.stage_failed,
                crash_result.error_message,
            )
            return CandidateResult(
                code=code,
                stage=EvalStage.CRASHED,
                crash_result=crash_result,
                eval_time=time.time() - t_start,
            )

        obs_dim = crash_result.obs_dim
        logger.info("Crash filter passed (obs_dim=%s)", obs_dim)

        # Load the interface for training
        interface = MDPInterface.from_code(code, required_functions=required_functions)
        interface.obs_dim = obs_dim

        # Apply mode-specific overrides
        mode = self.config.evolution_mode
        if mode == "reward_only":
            # Use adapter's default observation
            default_obs = self.adapter.get_default_obs_fn()
            interface.get_observation = default_obs
            # Detect obs_dim from default obs
            import jax.numpy as jnp
            test_obs = default_obs(self.dummy_state)
            obs_dim = int(jnp.asarray(test_obs).shape[0])
            interface.obs_dim = obs_dim
        elif mode == "obs_only":
            # Use env's built-in reward (None signals wrapper to keep default)
            interface.compute_reward = self.adapter.get_default_reward_fn()

        # --- Stage 2: Short training ---
        if self.config.evaluator.cascade_evaluation:
            result = self._cascade_evaluate(
                code, interface, obs_dim, crash_result, t_start
            )
        else:
            result = self._full_evaluate(
                code, interface, obs_dim, crash_result, t_start
            )

        return result

    def _cascade_evaluate(
        self,
        code: str,
        interface: MDPInterface,
        obs_dim: int,
        crash_result: CrashFilterResult,
        t_start: float,
    ) -> CandidateResult:
        """Run cascade: short training → threshold → full training."""
        tc = self.config.training
        ec = self.config.evaluator
        threshold = ec.cascade_thresholds[0] if ec.cascade_thresholds else 0.0

        # Short training
        logger.info(
            "Running short training (%s steps)...", f"{tc.total_timesteps:,}"
        )
        try:
            short_metrics = run_training(
                self.config,
                self.adapter,
                interface,
                obs_dim,
                total_timesteps=tc.total_timesteps,
            )
        except Exception as e:
            logger.error("Short training failed: %s", e)
            return CandidateResult(
                code=code,
                stage=EvalStage.SHORT_TRAIN,
                crash_result=crash_result,
                obs_dim=obs_dim,
                metrics={"final_return": 0.0, "final_length": 0.0, "success_rate": 0.0, "error": str(e)},
                eval_time=time.time() - t_start,
            )

        short_success = short_metrics.get("success_rate", 0.0)
        logger.info(
            "Short training done: success=%.0f%%, length=%.1f, time=%.1fs",
            short_success * 100,
            short_metrics.get("final_length", 0.0),
            short_metrics.get("training_time", 0.0),
        )

        # Threshold check (on success rate)
        if short_success < threshold:
            logger.info(
                "Short training success %.0f%% < threshold %.0f%% — rejected",
                short_success * 100,
                threshold * 100,
            )
            return CandidateResult(
                code=code,
                stage=EvalStage.SHORT_TRAIN_REJECTED,
                crash_result=crash_result,
                obs_dim=obs_dim,
                metrics=short_metrics,
                eval_time=time.time() - t_start,
            )

        # Full training
        logger.info(
            "Passed threshold (%.0f%% >= %.0f%%). Running full training (%s steps)...",
            short_success * 100,
            threshold * 100,
            f"{tc.total_timesteps_full:,}",
        )
        try:
            full_metrics = run_training(
                self.config,
                self.adapter,
                interface,
                obs_dim,
                total_timesteps=tc.total_timesteps_full,
            )
        except Exception as e:
            logger.error("Full training failed: %s", e)
            # Fall back to short training metrics
            return CandidateResult(
                code=code,
                stage=EvalStage.SHORT_TRAIN,
                crash_result=crash_result,
                obs_dim=obs_dim,
                metrics=short_metrics,
                eval_time=time.time() - t_start,
            )

        logger.info(
            "Full training done: success=%.0f%%, length=%.1f, time=%.1fs",
            full_metrics.get("success_rate", 0.0) * 100,
            full_metrics.get("final_length", 0.0),
            full_metrics.get("training_time", 0.0),
        )

        return CandidateResult(
            code=code,
            stage=EvalStage.FULL_TRAIN,
            crash_result=crash_result,
            obs_dim=obs_dim,
            metrics=full_metrics,
            eval_time=time.time() - t_start,
        )

    def _full_evaluate(
        self,
        code: str,
        interface: MDPInterface,
        obs_dim: int,
        crash_result: CrashFilterResult,
        t_start: float,
    ) -> CandidateResult:
        """Skip cascade, go directly to full training."""
        tc = self.config.training

        logger.info(
            "Running full training (%s steps, no cascade)...",
            f"{tc.total_timesteps_full:,}",
        )
        try:
            metrics = run_training(
                self.config,
                self.adapter,
                interface,
                obs_dim,
                total_timesteps=tc.total_timesteps_full,
            )
        except Exception as e:
            logger.error("Full training failed: %s", e)
            return CandidateResult(
                code=code,
                stage=EvalStage.FULL_TRAIN,
                crash_result=crash_result,
                obs_dim=obs_dim,
                metrics={"final_return": 0.0, "final_length": 0.0, "success_rate": 0.0, "error": str(e)},
                eval_time=time.time() - t_start,
            )

        logger.info(
            "Full training done: success=%.0f%%, length=%.1f, time=%.1fs",
            metrics.get("success_rate", 0.0) * 100,
            metrics.get("final_length", 0.0),
            metrics.get("training_time", 0.0),
        )

        return CandidateResult(
            code=code,
            stage=EvalStage.FULL_TRAIN,
            crash_result=crash_result,
            obs_dim=obs_dim,
            metrics=metrics,
            eval_time=time.time() - t_start,
        )
