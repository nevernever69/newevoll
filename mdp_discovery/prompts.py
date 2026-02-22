"""Prompt templates and builder for MDP Interface Discovery.

Assembles system/user message pairs for two scenarios:
- From scratch: first generation with no parent (iteration 1 or fresh island)
- Evolutionary: improve on a parent program using feedback (errors, metrics, top programs)
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from mdp_discovery.config import PromptConfig
from mdp_discovery.crash_filter import CrashFilterResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_NAMES: Dict[int, str] = {
    0: "Syntax Check (stage 0) — code could not be parsed",
    1: "Import Check (stage 1) — module failed to load",
    2: "Dry-Run (stage 2) — runtime error when calling functions with a real environment state",
}

# Extraction patterns, ordered by specificity (most → least specific).
_FENCE_PYTHON_RE = re.compile(r"```[Pp]ython\s*\n(.*?)```", re.DOTALL)
_FENCE_ANY_RE = re.compile(r"```\w*\s*\n(.*?)```", re.DOTALL)
_BARE_CODE_RE = re.compile(
    r"(import jax\b.*?(?:def get_observation|def compute_reward).*)",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Mode-dependent content helpers
# ---------------------------------------------------------------------------

_FUNCTION_DESC = {
    "full": (
        "Your task is to write two JAX-compatible Python functions:\n"
        "1. `get_observation(state)` — extracts a 1D float32 observation vector from the environment state\n"
        "2. `compute_reward(state, action, next_state)` — computes a scalar float32 reward signal"
    ),
    "reward_only": (
        "Your task is to write one JAX-compatible Python function:\n"
        "- `compute_reward(state, action, next_state)` — computes a scalar float32 reward signal\n\n"
        "The observation function is fixed (provided by the system). You only design the reward."
    ),
    "obs_only": (
        "Your task is to write one JAX-compatible Python function:\n"
        "- `get_observation(state)` — extracts a 1D float32 observation vector from the environment state\n\n"
        "The reward function uses the environment's built-in reward. You only design the observation."
    ),
    "random": (
        "Your task is to write two JAX-compatible Python functions:\n"
        "1. `get_observation(state)` — extracts a 1D float32 observation vector from the environment state\n"
        "2. `compute_reward(state, action, next_state)` — computes a scalar float32 reward signal"
    ),
}

_OUTPUT_CONSTRAINTS = {
    "full": (
        "The code must define both `get_observation` and `compute_reward` with the exact "
        "signatures shown above. You may define helper constants or functions above them.\n"
        "Only `import jax` and `import jax.numpy as jnp` are allowed."
    ),
    "reward_only": (
        "The code must define `compute_reward(state, action, next_state)` returning a scalar jnp.float32.\n"
        "You may define helper constants or functions above it.\n"
        "Only `import jax` and `import jax.numpy as jnp` are allowed."
    ),
    "obs_only": (
        "The code must define `get_observation(state)` returning a 1D jnp.float32 array (max 512 elements).\n"
        "You may define helper constants or functions above it.\n"
        "Only `import jax` and `import jax.numpy as jnp` are allowed."
    ),
    "random": (
        "The code must define both `get_observation` and `compute_reward` with the exact "
        "signatures shown above. You may define helper constants or functions above them.\n"
        "Only `import jax` and `import jax.numpy as jnp` are allowed."
    ),
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert reinforcement learning engineer. You design MDP interfaces \
for RL agents in reinforcement learning environments.

{function_description}

The RL agent sees ONLY what get_observation returns and is trained ONLY on what \
compute_reward provides. Your design of these functions determines whether the \
agent can learn the task.

## Design Philosophy

**The goal is to help a neural network LEARN, not to solve the task yourself.**

Observation design:
- The observation must give the agent ALL the information it needs to learn \
the task. Think carefully about what the agent needs to perceive from the \
environment state.
- Include any relational or structural information that is relevant to the task.
- Normalize features to similar ranges (e.g., [0, 1] or [-1, 1]).
- The observation can be anywhere from a handful of features to hundreds — \
use whatever the task requires. The max is 512 elements.
- Compute everything from the current state. Do NOT hardcode any environment-\
specific constants — the environment is randomized across episodes.

Reward design:
- The reward should provide a learning signal that guides the agent toward \
completing the task. The agent is evaluated on task success rate (% of episodes \
where the goal is reached), NOT on reward magnitude.
- Consider what sub-objectives or milestones exist in the task, and whether \
rewarding progress toward them would help learning.
- The reward can use any structure — dense shaping, sparse bonuses, \
milestone rewards, or combinations. Use what fits the task.

Generalization:
- The environment is evaluated across RANDOMIZED initial conditions. Your \
interface must generalize across all of them, not just one specific setup.
- Do NOT hardcode environment-specific constants. Compute everything from \
the current state.
- The reward must reflect ACTUAL task progress, not a proxy that happens to \
correlate in some configurations. If the reward can be maximized without \
completing the task, the agent will learn to exploit that shortcut.
- Test your reasoning: would this observation and reward still make sense if \
the initial conditions were completely different?

## Library Reference

{library_context}

## Output Format

Return your code inside a single Python markdown fence (```python ... ```).
{output_constraints}"""

USER_PROMPT_FROM_SCRATCH = """\
## Task

{task_description}

{failure_section}

## Instructions

{function_instructions}

{obs_instructions}\
{reward_instructions}\
JAX pitfalls to avoid:
- jax.lax.select requires both branches to have the SAME dtype (cast explicitly)
- Array indices must be integers, not floats (use .astype(jnp.int32))
- No Python if/else on JAX arrays (use jax.lax.select or jnp.where)
- Cast integer state values to float32 before arithmetic

Constraints:
{constraints}
- All code must be JAX-traceable (no Python if/else on JAX arrays)
- Only jax and jax.numpy imports are allowed"""

USER_PROMPT_EVOLUTIONARY = """\
## Task

{task_description}

## Parent Program

The following program is the starting point for your improvement:

{parent_section}

{feedback_sections}

## Instructions

Based on the feedback above, write an improved version of the MDP interface.

{improvement_guidance}

Constraints:
{constraints}
- All code must be JAX-traceable (no Python if/else on JAX arrays)
- Only jax and jax.numpy imports are allowed

**Output your improved code inside a single ```python ... ``` fence. No text outside the fence.**"""

# ---------------------------------------------------------------------------
# Section templates
# ---------------------------------------------------------------------------

SECTION_PARENT_CODE = """\
--- BEGIN CODE ---
{code}
--- END CODE ---
Success rate: {success_rate}  |  Episode length: {episode_length}"""

SECTION_PARENT_CODE_CRASHED = """\
--- BEGIN CODE ---
{code}
--- END CODE ---
(This program crashed before training — see failures below.)"""

SECTION_TOP_PROGRAMS = """\
## Best Known Programs

{entries}"""

SECTION_TOP_PROGRAM_ENTRY = """\
### Program #{rank} (success rate: {success_rate})
--- BEGIN CODE ---
{code}
--- END CODE ---"""

SECTION_DIVERSE_PROGRAMS = """\
## Diverse Approaches (different strategies worth considering)

{entries}"""

SECTION_DIVERSE_PROGRAM_ENTRY = """\
### Approach #{rank} (success rate: {success_rate})
--- BEGIN CODE ---
{code}
--- END CODE ---"""

SECTION_FAILURES = """\
## Recent Failures (avoid these mistakes)

{entries}"""

SECTION_FAILURE_CRASH = """\
### Failed Program #{rank}
--- BEGIN CODE ---
{code}
--- END CODE ---
**Crashed at: {stage_name}**
Error: {error_message}
Traceback (last {tb_lines} lines):
{traceback}"""

SECTION_FAILURE_LOW_FITNESS = """\
### Low-Performing Program #{rank} (success rate: {success_rate})
--- BEGIN CODE ---
{code}
--- END CODE ---
This program ran but the agent rarely or never reached the goal."""

SECTION_TRAINING_FEEDBACK = """\
## Training Feedback for Parent

{analysis}"""

SECTION_INSPIRATION_PROGRAMS = """\
## Inspiration Programs (structurally different strategies)

Consider borrowing ideas from these, even if their success rates differ.

{entries}"""

SECTION_INSPIRATION_PROGRAM_ENTRY = """\
### Alternative Strategy #{rank} (success rate: {success_rate})
--- BEGIN CODE ---
{code}
--- END CODE ---"""


# ---------------------------------------------------------------------------
# Improvement guidance variants (for prompt stochasticity)
# ---------------------------------------------------------------------------

_IMPROVEMENT_VARIANTS = {
    "zero": [
        # Strategy A: rethink observation — maybe the agent is blind
        (
            "Zero success — the interface is not working at all.\n"
            "Focus on the OBSERVATION: the agent may lack the information "
            "it needs to learn. Consider what state features are critical "
            "for the task and whether the agent can perceive them. "
            "Keep the reward simple and revisit it later."
        ),
        # Strategy B: rethink reward — maybe the agent has no learning signal
        (
            "Zero success — the interface is not working at all.\n"
            "Focus on the REWARD: the agent may have no useful learning "
            "signal. Consider whether the reward is too sparse or always "
            "near-zero. A denser reward that changes on every step — even "
            "approximately — may be needed. The observation might be fine."
        ),
        # Strategy C: scrap both, try something structurally opposite
        (
            "Zero success — the interface is not working at all.\n"
            "Try the OPPOSITE approach to what the parent did. If the "
            "observation was complex, try minimal. If the reward was dense, "
            "try sparse milestones. If both were simple, try richer "
            "representations. Break out of the current design direction."
        ),
    ],
    "low": [
        # Strategy A: observation is the bottleneck
        (
            "Low success rate — the agent struggles to learn.\n"
            "The observation may be the bottleneck. Is it missing "
            "information the agent needs? Does it contain redundant or "
            "misleading features? Try changing what the agent perceives."
        ),
        # Strategy B: reward is the bottleneck
        (
            "Low success rate — the agent struggles to learn.\n"
            "The reward may be the bottleneck. Are reward components "
            "conflicting with each other? Is the magnitude appropriate? "
            "Try simplifying the reward to a single clear signal."
        ),
        # Strategy C: representation mismatch
        (
            "Low success rate — the agent struggles to learn.\n"
            "The observation and reward may be individually reasonable but "
            "mismatched. Does the reward incentivize behavior that the "
            "observation can actually distinguish? Ensure they work together."
        ),
    ],
    "medium": [
        # Strategy A: diagnose the failure mode
        (
            "Moderate success — partially working.\n"
            "Identify what causes the remaining failures. Is the agent "
            "failing in specific states? Add observation features or reward "
            "signals that address those failure modes specifically."
        ),
        # Strategy B: simplify
        (
            "Moderate success — partially working.\n"
            "The design may be overfit to easy cases. Try simplifying — "
            "remove observation features or reward terms that might be "
            "adding noise, and see if a cleaner interface generalizes better."
        ),
        # Strategy C: enrich
        (
            "Moderate success — partially working.\n"
            "The agent may need more information to handle harder cases. "
            "Consider adding state features it currently cannot perceive, "
            "or adding reward terms for sub-goals it currently ignores."
        ),
    ],
    "good": [
        # Strategy A: analyze failure margin
        (
            "Good performance with room to improve.\n"
            "Focus on the remaining failure cases. What distinguishes them "
            "from successes? Target those specific scenarios with observation "
            "or reward adjustments."
        ),
        # Strategy B: reduce and stabilize
        (
            "Good performance with room to improve.\n"
            "Consider whether the interface can be simplified without "
            "losing performance. Fewer observation features or simpler "
            "reward logic may improve learning stability."
        ),
        # Strategy C: push reward refinement
        (
            "Good performance with room to improve.\n"
            "The observation is likely sufficient. Focus on whether the "
            "reward signal has the right relative magnitudes across its "
            "components, and whether it properly incentivizes completing "
            "the task efficiently."
        ),
    ],
    "excellent": [
        # Strategy A: polish edges
        (
            "Excellent performance. Small refinements only.\n"
            "Analyze the rare failure cases and make targeted changes. "
            "Avoid large structural changes that might destabilize what works."
        ),
        # Strategy B: compress
        (
            "Excellent performance. Consider compressing the interface.\n"
            "Can the observation dimensionality be reduced without losing "
            "performance? Can the reward be simplified? A simpler interface "
            "that maintains this success rate is strictly better."
        ),
        # Strategy C: robustness
        (
            "Excellent performance. Focus on robustness.\n"
            "The remaining failures may be edge cases in the environment. "
            "Check if the observation handles all possible states correctly "
            "and if the reward avoids any degenerate values."
        ),
    ],
}


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from the LLM response.

    Multi-tier extraction (most specific → least):
      1. ```python ... ``` fence
      2. Any ``` ... ``` fence whose content looks like Python
      3. Bare code starting with ``import jax`` (no fence at all)
    Returns None only if all tiers fail.
    """
    # Tier 1: explicit ```python fence
    match = _FENCE_PYTHON_RE.search(response)
    if match:
        return match.group(1).strip()

    # Tier 2: any ``` fence containing Python-ish code
    for match in _FENCE_ANY_RE.finditer(response):
        code = match.group(1).strip()
        if "import jax" in code or "def get_observation" in code or "def compute_reward" in code:
            return code

    # Tier 3: bare code (model forgot the fence entirely)
    match = _BARE_CODE_RE.search(response)
    if match:
        return match.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Builds prompts for the LLM generation engine.

    Usage:
        builder = PromptBuilder(config, library_context, task_description,
                                evolution_mode="full")

        # From scratch (iteration 1 or new island):
        prompt = builder.build_prompt()

        # Evolutionary (improve on parent):
        prompt = builder.build_prompt(
            parent_code=parent.source_code,
            parent_metrics={"success_rate": 0.5, "final_length": 95, ...},
            failed_programs=[{"code": "...", "crash_result": CrashFilterResult(...)}],
            top_programs=[{"code": "...", "metrics": {...}}],
            best_metrics={"success_rate": 0.8, ...},
        )

        # Send to LLM:
        response = llm.chat(system=prompt["system"], user=prompt["user"])
        code = extract_code(response)
    """

    def __init__(
        self,
        config: PromptConfig,
        library_context: str,
        task_description: str,
        evolution_mode: str = "full",
    ):
        self.config = config
        self.task_description = task_description
        self.evolution_mode = evolution_mode

        # Build mode-dependent content
        function_description = _FUNCTION_DESC.get(
            evolution_mode, _FUNCTION_DESC["full"]
        )
        output_constraints = _OUTPUT_CONSTRAINTS.get(
            evolution_mode, _OUTPUT_CONSTRAINTS["full"]
        )

        self._system_prompt = SYSTEM_PROMPT.format(
            function_description=function_description,
            library_context=library_context,
            output_constraints=output_constraints,
        )

    def _get_obs_instructions(self) -> str:
        """Return observation instructions based on mode."""
        if self.evolution_mode == "reward_only":
            return ""
        return (
            "**Observation design:**\n"
            "- Decide what information from the environment state the agent needs "
            "to solve this task.\n"
            "- Compute all features from the current state — do NOT hardcode "
            "any environment-specific constants.\n"
            "- Refer to the library reference above for all available state fields.\n\n"
        )

    def _get_reward_instructions(self) -> str:
        """Return reward instructions based on mode."""
        if self.evolution_mode == "obs_only":
            return ""
        return (
            "**Reward design:**\n"
            "- Design a reward signal that will help the agent learn the task. "
            "Consider what milestones, progress measures, or completion signals "
            "are appropriate for this task.\n"
            "- The agent is evaluated on task success rate, not reward magnitude.\n"
            "- AVOID reward hacking: do not reward proxies that can be maximized "
            "without actually completing the task. The reward must track real progress.\n\n"
        )

    def _get_function_instructions(self) -> str:
        """Return top-level function instructions based on mode."""
        if self.evolution_mode == "reward_only":
            return (
                "Design a `compute_reward` function that will help a PPO "
                "agent learn the task described above.\n"
                "The observation is handled by the system — you only need to design the reward."
            )
        elif self.evolution_mode == "obs_only":
            return (
                "Design a `get_observation` function that will give a PPO agent "
                "the information it needs to learn the task described above.\n"
                "The reward uses the environment's built-in signal — you only need to design the observation."
            )
        else:
            return (
                "Design `get_observation` and `compute_reward` functions that will allow a PPO "
                "agent to learn the task described above."
            )

    def _get_constraints(self) -> str:
        """Return constraint lines based on mode."""
        lines = []
        if self.evolution_mode != "reward_only":
            lines.append("- get_observation must return a 1D jnp.float32 array (max 512 elements)")
        if self.evolution_mode != "obs_only":
            lines.append("- compute_reward must return a scalar jnp.float32")
        lines.append(
            "- Do NOT hardcode environment-specific constants. "
            "The environment is randomized — the interface must generalize"
        )
        if self.evolution_mode != "obs_only":
            lines.append(
                "- The reward must reflect actual task progress. Do not reward proxies "
                "that can be maximized without completing the task (reward hacking)"
            )
        return "\n".join(lines)

    def build_prompt(
        self,
        *,
        parent_code: Optional[str] = None,
        parent_metrics: Optional[Dict[str, Any]] = None,
        top_programs: Optional[List[Dict[str, Any]]] = None,
        diverse_programs: Optional[List[Dict[str, Any]]] = None,
        failed_programs: Optional[List[Dict[str, Any]]] = None,
        best_metrics: Optional[Dict[str, Any]] = None,
        inspiration_programs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """Build a system/user prompt pair for the LLM.

        Args:
            parent_code: Source code of the parent program. None = from-scratch.
            parent_metrics: Training metrics of the parent (from run_training).
            top_programs: Best programs from the population.
                Each: {"code": str, "metrics": {"final_return": float, ...}}
            diverse_programs: Programs from different MAP-Elites cells.
                Same format as top_programs.
            failed_programs: Recent failures to learn from.
                Each: {"code": str, "crash_result": CrashFilterResult} for crashes,
                or {"code": str, "metrics": {"final_return": float, ...}} for low-fitness.
            best_metrics: Metrics of the overall best program, for comparison.
            inspiration_programs: Structurally distant programs for exploration.
                Same format as top_programs.

        Returns:
            {"system": str, "user": str} ready for the LLM API.
        """
        if parent_code is None:
            user_msg = self._build_from_scratch(failed_programs=failed_programs)
        else:
            user_msg = self._build_evolutionary(
                parent_code=parent_code,
                parent_metrics=parent_metrics,
                top_programs=top_programs,
                diverse_programs=diverse_programs,
                failed_programs=failed_programs,
                best_metrics=best_metrics,
                inspiration_programs=inspiration_programs,
            )
        return {"system": self._system_prompt, "user": user_msg}

    # ------------------------------------------------------------------
    # From-scratch
    # ------------------------------------------------------------------

    def _build_from_scratch(
        self, failed_programs: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        failure_section = ""
        if failed_programs:
            failure_section = self._format_failures(
                failed_programs[: self.config.num_failure_examples]
            )
        return USER_PROMPT_FROM_SCRATCH.format(
            task_description=self.task_description,
            failure_section=failure_section,
            function_instructions=self._get_function_instructions(),
            obs_instructions=self._get_obs_instructions(),
            reward_instructions=self._get_reward_instructions(),
            constraints=self._get_constraints(),
        )

    # ------------------------------------------------------------------
    # Evolutionary
    # ------------------------------------------------------------------

    def _build_evolutionary(
        self,
        *,
        parent_code: str,
        parent_metrics: Optional[Dict[str, Any]],
        top_programs: Optional[List[Dict[str, Any]]],
        diverse_programs: Optional[List[Dict[str, Any]]],
        failed_programs: Optional[List[Dict[str, Any]]],
        best_metrics: Optional[Dict[str, Any]],
        inspiration_programs: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        stochastic = self.config.use_stochasticity

        # Parent section
        parent_section = self._format_parent(parent_code, parent_metrics)

        # Feedback sections (only non-empty ones)
        sections: List[str] = []

        if failed_programs:
            s = self._format_failures(
                failed_programs[: self.config.num_failure_examples]
            )
            if s:
                sections.append(s)

        if top_programs:
            progs = list(top_programs[: self.config.num_top_programs])
            if stochastic:
                random.shuffle(progs)
            s = self._format_top_programs(progs)
            if s:
                sections.append(s)

        if diverse_programs:
            # Stochastic: 70% chance of including diverse programs
            include_diverse = not stochastic or random.random() < 0.7
            if include_diverse:
                progs = list(diverse_programs[: self.config.num_diverse_programs])
                if stochastic:
                    random.shuffle(progs)
                s = self._format_diverse_programs(progs)
                if s:
                    sections.append(s)

        if inspiration_programs:
            s = self._format_inspiration_programs(inspiration_programs)
            if s:
                sections.append(s)

        if parent_metrics and best_metrics:
            # Stochastic: 80% chance of including training feedback
            include_feedback = not stochastic or random.random() < 0.8
            if include_feedback:
                s = self._format_training_feedback(parent_metrics, best_metrics)
                if s:
                    sections.append(s)

        feedback_sections = "\n\n".join(sections)

        # Improvement guidance
        improvement_guidance = self._build_improvement_guidance(
            parent_metrics, failed_programs, best_metrics
        )

        return USER_PROMPT_EVOLUTIONARY.format(
            task_description=self.task_description,
            parent_section=parent_section,
            feedback_sections=feedback_sections,
            improvement_guidance=improvement_guidance,
            constraints=self._get_constraints(),
        )

    # ------------------------------------------------------------------
    # Section formatters
    # ------------------------------------------------------------------

    def _format_parent(
        self, code: str, metrics: Optional[Dict[str, Any]]
    ) -> str:
        if metrics:
            sr = metrics.get("success_rate", 0.0)
            return SECTION_PARENT_CODE.format(
                code=code,
                success_rate=f"{sr:.0%}",
                episode_length=f"{metrics.get('final_length', 0.0):.1f}",
            )
        return SECTION_PARENT_CODE_CRASHED.format(code=code)

    def _format_top_programs(self, programs: List[Dict[str, Any]]) -> str:
        if not programs:
            return ""
        entries = []
        for i, prog in enumerate(programs, 1):
            m = prog.get("metrics", {})
            sr = m.get("success_rate", 0.0)
            entries.append(
                SECTION_TOP_PROGRAM_ENTRY.format(
                    rank=i,
                    success_rate=f"{sr:.0%}",
                    code=prog["code"],
                )
            )
        return SECTION_TOP_PROGRAMS.format(entries="\n\n".join(entries))

    def _format_diverse_programs(self, programs: List[Dict[str, Any]]) -> str:
        if not programs:
            return ""
        entries = []
        for i, prog in enumerate(programs, 1):
            m = prog.get("metrics", {})
            sr = m.get("success_rate", 0.0)
            entries.append(
                SECTION_DIVERSE_PROGRAM_ENTRY.format(
                    rank=i,
                    success_rate=f"{sr:.0%}",
                    code=prog["code"],
                )
            )
        return SECTION_DIVERSE_PROGRAMS.format(entries="\n\n".join(entries))

    def _format_inspiration_programs(self, programs: List[Dict[str, Any]]) -> str:
        if not programs:
            return ""
        entries = []
        for i, prog in enumerate(programs, 1):
            m = prog.get("metrics", {})
            sr = m.get("success_rate", 0.0)
            entries.append(
                SECTION_INSPIRATION_PROGRAM_ENTRY.format(
                    rank=i,
                    success_rate=f"{sr:.0%}",
                    code=prog["code"],
                )
            )
        return SECTION_INSPIRATION_PROGRAMS.format(entries="\n\n".join(entries))

    def _format_failures(self, failures: List[Dict[str, Any]]) -> str:
        if not failures:
            return ""
        entries = []
        for i, fail in enumerate(failures, 1):
            code = fail["code"]
            crash: Optional[CrashFilterResult] = fail.get("crash_result")

            if crash is not None and not crash.passed:
                entries.append(self._format_crash_error(i, code, crash))
            else:
                m = fail.get("metrics", {})
                sr = m.get("success_rate", 0.0)
                entries.append(
                    SECTION_FAILURE_LOW_FITNESS.format(
                        rank=i,
                        success_rate=f"{sr:.0%}",
                        code=code,
                    )
                )
        return SECTION_FAILURES.format(entries="\n\n".join(entries))

    def _format_crash_error(
        self, rank: int, code: str, crash: CrashFilterResult
    ) -> str:
        stage_name = STAGE_NAMES.get(
            crash.stage_failed, f"Unknown stage {crash.stage_failed}"
        )
        raw_tb = crash.error_traceback or ""
        traceback_text = self._truncate_traceback(raw_tb, max_lines=15)
        tb_lines = min(15, len(raw_tb.strip().splitlines()))

        return SECTION_FAILURE_CRASH.format(
            rank=rank,
            code=code,
            stage_name=stage_name,
            error_message=crash.error_message or "Unknown error",
            tb_lines=tb_lines if tb_lines > 0 else "N/A",
            traceback=traceback_text,
        )

    def _format_training_feedback(
        self,
        metrics: Dict[str, Any],
        best_metrics: Dict[str, Any],
    ) -> str:
        success_rate = metrics.get("success_rate", 0.0)
        best_success = best_metrics.get("success_rate", 0.0)
        episode_length = metrics.get("final_length", 0.0)

        parts: List[str] = []

        parts.append(
            f"- Parent success rate: {success_rate:.0%} "
            f"(mean episode length: {episode_length:.1f})"
        )
        parts.append(f"- Best known success rate: {best_success:.0%}")

        if success_rate == 0:
            parts.append(
                "- The agent NEVER succeeded. The current interface is not "
                "producing a useful learning signal. The observation, the reward, "
                "or both need significant changes."
            )
        elif success_rate < 0.3:
            parts.append(
                "- The agent rarely succeeds. The interface is limiting learning. "
                "Consider whether the observation provides enough information, "
                "or whether the reward signal is misleading or too weak."
            )
        elif success_rate < 0.6:
            parts.append(
                "- The agent succeeds sometimes but not reliably. The interface "
                "is partially working. Consider what causes the remaining "
                "failures — missing observation features? Conflicting reward "
                "signals? Unhandled edge cases?"
            )
        elif success_rate < 0.9:
            parts.append(
                "- Good success rate with room to improve. Focus on what causes "
                "the remaining failures."
            )

        # Plateau detection
        if best_success > 0 and abs(success_rate - best_success) < 0.05:
            parts.append(
                "- NOTE: The parent is near the best known success rate, "
                "suggesting the current approach may be plateauing. Consider "
                "trying a structurally different interface design."
            )

        # Success curve trend
        curve = metrics.get("success_curve", [])
        if len(curve) >= 3:
            third = max(len(curve) // 3, 1)
            early = sum(curve[:third]) / third
            late = sum(curve[-third:]) / third
            if late > early + 0.1:
                parts.append(
                    "- Learning curve is still improving at the end of "
                    "training — the interface may benefit from more training "
                    "time, or could be refined to converge faster."
                )
            elif late < early - 0.1:
                parts.append(
                    "- Learning curve is DECLINING — training is unstable. "
                    "The reward signal may be causing optimization issues."
                )
            elif late < 0.1:
                parts.append(
                    "- Success rate stays near zero throughout training. "
                    "The current interface is not working — try something "
                    "structurally different."
                )

        # Generalization reminder
        parts.append(
            "- IMPORTANT: The environment is randomized across episodes. "
            "Ensure the interface does not rely on hardcoded constants. "
            "The reward must reflect real task completion, not a proxy "
            "that can be maximized without solving the task."
        )

        analysis = "\n".join(parts)
        return SECTION_TRAINING_FEEDBACK.format(analysis=analysis)

    def _build_improvement_guidance(
        self,
        parent_metrics: Optional[Dict[str, Any]],
        failed_programs: Optional[List[Dict[str, Any]]],
        best_metrics: Optional[Dict[str, Any]],
    ) -> str:
        hints: List[str] = []
        stochastic = self.config.use_stochasticity

        # Check for crash failures
        has_crashes = False
        if failed_programs:
            for f in failed_programs:
                cr: Optional[CrashFilterResult] = f.get("crash_result")
                if cr and not cr.passed:
                    has_crashes = True
                    break

        if has_crashes:
            hints.append(
                "Some recent attempts crashed. Study the error messages "
                "carefully and ensure your code is JAX-traceable. Common "
                "issues:\n"
                "  - Using Python if/else on JAX arrays (use jax.lax.select "
                "instead)\n"
                "  - Wrong attribute access on the state object\n"
                "  - Shape mismatches in jnp.concatenate\n"
                "  - Using numpy instead of jax.numpy"
            )

        if parent_metrics:
            sr = parent_metrics.get("success_rate", 0.0)
            best_sr = best_metrics.get("success_rate", 0.0) if best_metrics else 0.0

            if sr == 0:
                bracket = "zero"
            elif sr < 0.3:
                bracket = "low"
            elif sr < 0.6:
                bracket = "medium"
            elif sr < 0.9:
                bracket = "good"
            else:
                bracket = "excellent"

            variants = _IMPROVEMENT_VARIANTS[bracket]
            if stochastic:
                hints.append(random.choice(variants))
            else:
                hints.append(variants[0])

            # Plateau detection: parent ≈ best and best isn't great
            if best_sr > 0 and abs(sr - best_sr) < 0.1 and best_sr < 0.85:
                hints.append(
                    "**PLATEAU DETECTED**: The population has stagnated near "
                    f"{best_sr:.0%} success. Try a STRUCTURALLY DIFFERENT "
                    "interface design — different observation representation, "
                    "different reward structure, or both."
                )

        if not hints:
            hints.append(
                "Think carefully about what information the agent needs to "
                "solve this task (observation) and what learning signal will "
                "guide it toward the goal (reward)."
            )

        return "\n\n".join(hints)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_traceback(tb: str, max_lines: int = 15) -> str:
        """Keep only the last max_lines of a traceback (most informative)."""
        lines = tb.strip().splitlines()
        if len(lines) <= max_lines:
            return tb.strip()
        truncated = lines[-max_lines:]
        return "    ... (earlier frames omitted) ...\n" + "\n".join(truncated)
