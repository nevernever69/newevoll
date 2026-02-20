"""Evolution tracing — logs parent->child transitions to JSONL for analysis.

Records every evolutionary step with metrics deltas, enabling paper-quality
analysis of search dynamics, improvement rates, and model contributions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvolutionEvent:
    """A single parent->child evolutionary transition."""

    iteration: int
    timestamp: float
    parent_id: Optional[str]
    child_id: str
    parent_metrics: Optional[Dict[str, Any]]
    child_metrics: Dict[str, Any]
    improvement_delta: Dict[str, float]
    obs_dim: Optional[int] = None
    island: int = 0
    generation: int = 0
    stage: str = ""
    model: str = ""
    parent_code: Optional[str] = None
    child_code: Optional[str] = None
    prompt_user: Optional[str] = None
    llm_response_text: Optional[str] = None


def _compute_improvement(
    parent_metrics: Optional[Dict[str, Any]],
    child_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Compute numeric deltas between parent and child metrics."""
    if parent_metrics is None:
        return {}
    delta: Dict[str, float] = {}
    for key, child_val in child_metrics.items():
        if not isinstance(child_val, (int, float)):
            continue
        parent_val = parent_metrics.get(key)
        if isinstance(parent_val, (int, float)):
            delta[key] = float(child_val) - float(parent_val)
    return delta


class EvolutionTracer:
    """Buffered JSONL writer for evolution events.

    Usage:
        tracer = EvolutionTracer(output_path, ...)
        tracer.log_event(iteration=1, parent_id="abc", ...)
        tracer.close()  # flushes + prints summary
    """

    def __init__(
        self,
        output_path: Path,
        include_code: bool = False,
        include_prompts: bool = False,
        buffer_size: int = 10,
        enabled: bool = True,
    ):
        self.output_path = Path(output_path)
        self.include_code = include_code
        self.include_prompts = include_prompts
        self.buffer_size = buffer_size
        self.enabled = enabled

        self._buffer: List[Dict[str, Any]] = []

        # Running stats
        self.total_events = 0
        self.improvement_count = 0
        self.best_improvement: Dict[str, float] = {}

    def log_event(
        self,
        *,
        iteration: int,
        parent_id: Optional[str],
        child_id: str,
        parent_metrics: Optional[Dict[str, Any]],
        child_metrics: Dict[str, Any],
        obs_dim: Optional[int] = None,
        island: int = 0,
        generation: int = 0,
        stage: str = "",
        model: str = "",
        parent_code: Optional[str] = None,
        child_code: Optional[str] = None,
        prompt_user: Optional[str] = None,
        llm_response_text: Optional[str] = None,
    ) -> None:
        """Record an evolution event."""
        if not self.enabled:
            return

        improvement_delta = _compute_improvement(parent_metrics, child_metrics)

        event = EvolutionEvent(
            iteration=iteration,
            timestamp=time.time(),
            parent_id=parent_id,
            child_id=child_id,
            parent_metrics=parent_metrics,
            child_metrics=child_metrics,
            improvement_delta=improvement_delta,
            obs_dim=obs_dim,
            island=island,
            generation=generation,
            stage=stage,
            model=model,
            parent_code=parent_code if self.include_code else None,
            child_code=child_code if self.include_code else None,
            prompt_user=prompt_user if self.include_prompts else None,
            llm_response_text=llm_response_text if self.include_prompts else None,
        )

        # Update running stats
        self.total_events += 1
        sr_delta = improvement_delta.get("success_rate")
        if sr_delta is not None and sr_delta > 0:
            self.improvement_count += 1
        for key, val in improvement_delta.items():
            if key not in self.best_improvement or val > self.best_improvement[key]:
                self.best_improvement[key] = val

        self._buffer.append(asdict(event))
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered events to JSONL file."""
        if not self.enabled or not self._buffer:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a") as f:
            for event_dict in self._buffer:
                f.write(json.dumps(event_dict, default=str) + "\n")

        logger.debug("Flushed %d trace events to %s", len(self._buffer), self.output_path)
        self._buffer.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            "total_events": self.total_events,
            "improvement_count": self.improvement_count,
            "improvement_rate": (
                self.improvement_count / self.total_events
                if self.total_events > 0
                else 0.0
            ),
            "best_improvement": dict(self.best_improvement),
        }

    def close(self) -> None:
        """Flush remaining events and log summary."""
        self.flush()
        if self.enabled and self.total_events > 0:
            summary = self.get_summary()
            logger.info(
                "Evolution trace: %d events, %d improvements (%.0f%%), best deltas: %s",
                summary["total_events"],
                summary["improvement_count"],
                summary["improvement_rate"] * 100,
                summary["best_improvement"],
            )
