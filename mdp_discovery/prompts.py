"""Prompt templates and builder for MDP Interface Discovery.

Assembles system/user message pairs for two scenarios:
- From scratch: first generation with no parent (iteration 1 or fresh island)
- Evolutionary: improve on a parent program using feedback (errors, metrics, top programs)
"""

from __future__ import annotations

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

_CODE_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert reinforcement learning engineer. You design MDP interfaces \
for RL agents in grid-world environments.

Your task is to write two JAX-compatible Python functions:
1. `get_observation(state)` — extracts a 1D float32 observation vector from the environment state
2. `compute_reward(state, action, next_state)` — computes a scalar float32 reward signal

The RL agent sees ONLY what get_observation returns and is trained ONLY on what \
compute_reward provides. Your design of these functions determines whether the \
agent can learn the task.

## Design Philosophy

**The goal is to help a neural network LEARN, not to solve the task yourself.**

Observation design:
- The observation should give the neural network the RAW INFORMATION it needs \
to learn a good policy — positions, distances, directions, grid contents.
- Do NOT pre-compute "action values", "optimal directions", "turn scores", \
or anything that tries to solve the RL problem inside the observation. That is \
the neural network's job. Pre-computing policy hints creates brittle, hard-to-learn \
features that often hurt more than they help.
- Simpler observations with fewer, well-normalized features often outperform \
complex ones. A small observation trains faster and generalizes better.
- Normalize all features to similar ranges (e.g., [0, 1] or [-1, 1]).

Reward design:
- The reward should provide a SMOOTH LEARNING SIGNAL that guides the agent \
toward the goal. The agent is evaluated on whether it actually reaches the \
goal, NOT on reward magnitude.
- Distance-based shaping (reward for getting closer) is the most reliable \
dense signal for navigation tasks.
- Keep reward components few and well-scaled. Many competing reward terms \
with different magnitudes confuse learning.
- Do NOT inflate reward magnitudes to "hack" higher returns — the fitness \
metric is success rate (% of episodes reaching the goal), not total reward.

Training budget: The agent trains on a limited number of environment steps. \
Simpler designs that converge quickly are preferred over complex ones that \
need more training to learn.

## Library Reference

{library_context}

## Output Format

Return your code inside a single Python markdown fence (```python ... ```).
The code must define both `get_observation` and `compute_reward` with the exact \
signatures shown above. You may define helper constants or functions above them.
Only `import jax` and `import jax.numpy as jnp` are allowed."""

USER_PROMPT_FROM_SCRATCH = """\
## Task

{task_description}

{failure_section}

## Instructions

Design `get_observation` and `compute_reward` functions that will allow a PPO \
agent to learn the task described above.

**Observation design** — give the neural network what it needs to learn:
- Agent position (normalized), agent direction (one-hot)
- Goal/target position (normalized), distance to goal
- Keep it simple: 8-15 features is usually plenty. Every extra feature is \
another dimension the neural network must learn to interpret.
- Do NOT pre-compute "action scores" or "optimal directions" — let the NN learn the policy.

**Reward design** — shape learning, not the score:
- Distance-based shaping: reward for getting closer to the goal each step
- Small per-step penalty to encourage efficiency
- Keep reward scale moderate (e.g., distance improvement * 1.0, goal bonus * 5-10)
- Fewer reward components is better. 2-3 clean terms beat 6 competing ones.

Common JAX pitfalls to avoid:
- jax.lax.select requires both branches to have the SAME dtype (cast explicitly)
- Array indices must be integers, not floats (use .astype(jnp.int32))
- No Python if/else on JAX arrays (use jax.lax.select or jnp.where)
- state.grid values are uint8 — cast to float32 before arithmetic

Constraints:
- get_observation must return a 1D jnp.float32 array (max 512 elements)
- compute_reward must return a scalar jnp.float32
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

Key principles:
- The observation provides RAW information for the neural network to learn from — \
do NOT pre-compute policy decisions (action values, optimal turns) in the observation.
- The reward shapes LEARNING — the agent is evaluated on goal success rate, not reward magnitude.
- Simpler designs train faster with limited compute. Remove features/reward terms that \
don't clearly help.

Constraints:
- get_observation must return a 1D jnp.float32 array (max 512 elements)
- compute_reward must return a scalar jnp.float32
- All code must be JAX-traceable (no Python if/else on JAX arrays)
- Only jax and jax.numpy imports are allowed"""

# ---------------------------------------------------------------------------
# Section templates
# ---------------------------------------------------------------------------

SECTION_PARENT_CODE = """\
```python
{code}
```
Success rate: {success_rate}  |  Episode length: {episode_length}"""

SECTION_PARENT_CODE_CRASHED = """\
```python
{code}
```
(This program crashed before training — see failures below.)"""

SECTION_TOP_PROGRAMS = """\
## Best Known Programs

{entries}"""

SECTION_TOP_PROGRAM_ENTRY = """\
### Program #{rank} (success rate: {success_rate})
```python
{code}
```"""

SECTION_DIVERSE_PROGRAMS = """\
## Diverse Approaches (different strategies worth considering)

{entries}"""

SECTION_DIVERSE_PROGRAM_ENTRY = """\
### Approach #{rank} (success rate: {success_rate})
```python
{code}
```"""

SECTION_FAILURES = """\
## Recent Failures (avoid these mistakes)

{entries}"""

SECTION_FAILURE_CRASH = """\
### Failed Program #{rank}
```python
{code}
```
**Crashed at: {stage_name}**
Error: {error_message}
Traceback (last {tb_lines} lines):
```
{traceback}
```"""

SECTION_FAILURE_LOW_FITNESS = """\
### Low-Performing Program #{rank} (success rate: {success_rate})
```python
{code}
```
This program ran but the agent rarely or never reached the goal."""

SECTION_TRAINING_FEEDBACK = """\
## Training Feedback for Parent

{analysis}"""


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from a markdown fence in the LLM response.

    Searches for the first ```python ... ``` block and returns its content.
    Returns None if no fence is found.
    """
    match = _CODE_FENCE_RE.search(response)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Builds prompts for the LLM generation engine.

    Usage:
        builder = PromptBuilder(config, library_context, task_description)

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
    ):
        self.config = config
        self.task_description = task_description
        self._system_prompt = SYSTEM_PROMPT.format(
            library_context=library_context,
        )

    def build_prompt(
        self,
        *,
        parent_code: Optional[str] = None,
        parent_metrics: Optional[Dict[str, Any]] = None,
        top_programs: Optional[List[Dict[str, Any]]] = None,
        diverse_programs: Optional[List[Dict[str, Any]]] = None,
        failed_programs: Optional[List[Dict[str, Any]]] = None,
        best_metrics: Optional[Dict[str, Any]] = None,
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
    ) -> str:
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
            s = self._format_top_programs(
                top_programs[: self.config.num_top_programs]
            )
            if s:
                sections.append(s)

        if diverse_programs:
            s = self._format_diverse_programs(
                diverse_programs[: self.config.num_diverse_programs]
            )
            if s:
                sections.append(s)

        if parent_metrics and best_metrics:
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
                "- The agent NEVER reached the goal. The reward is not "
                "providing a useful learning signal. Try a simple, strong "
                "distance-based shaping reward (reward = prev_dist - curr_dist)."
            )
        elif success_rate < 0.3:
            parts.append(
                "- The agent rarely reaches the goal. The reward provides "
                "some signal but learning is too slow. Consider: is the "
                "observation too complex for the NN to learn from quickly? "
                "Is the reward pulling in conflicting directions?"
            )
        elif success_rate < 0.6:
            parts.append(
                "- The agent reaches the goal sometimes but not reliably. "
                "This may indicate the observation is missing key information, "
                "or the reward has components that sometimes mislead the agent."
            )
        elif success_rate < 0.9:
            parts.append(
                "- Good success rate, but there is room to improve. Focus on "
                "why the remaining episodes fail — does the agent get stuck "
                "in certain starting positions? Is the reward scale balanced?"
            )

        # Plateau detection
        if best_success > 0 and abs(success_rate - best_success) < 0.05:
            parts.append(
                "- NOTE: The parent is near the best known success rate, "
                "suggesting the current approach may be plateauing. Consider "
                "trying a STRUCTURALLY DIFFERENT design rather than minor "
                "tweaks — e.g., different observation features, a simpler "
                "reward, or a different distance metric."
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
                    "training — the design is good but needs more training "
                    "time, or could converge faster with a simpler observation."
                )
            elif late < early - 0.1:
                parts.append(
                    "- Learning curve is DECLINING — the reward is causing "
                    "unstable training. Reduce reward magnitude, remove "
                    "competing reward terms, or simplify."
                )
            elif late < 0.1:
                parts.append(
                    "- Success rate stays near zero throughout training. "
                    "The current design is not working — try something "
                    "fundamentally different."
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
            obs_dim = parent_metrics.get("obs_dim", 0)

            if sr == 0:
                hints.append(
                    "The agent never reaches the goal. Start with the "
                    "simplest possible design:\n"
                    "  - Observation: agent position + direction + goal "
                    "position + distance (~8 features)\n"
                    "  - Reward: (prev_distance - curr_distance) + small "
                    "goal bonus + tiny step penalty\n"
                    "Strip everything else away and see if learning starts."
                )
            elif sr < 0.3:
                hints.append(
                    "The agent learns slowly. Possible causes:\n"
                    "  - Observation has too many features — reduce to "
                    "the essential ones (position, direction, goal, distance)\n"
                    "  - Reward has too many competing terms — simplify to "
                    "distance shaping + goal bonus + step penalty\n"
                    "  - Reward magnitudes are unbalanced — keep all terms "
                    "in a similar range (0.01 to 1.0)"
                )
            elif sr < 0.6:
                hints.append(
                    "Moderate success — the design is on the right track. "
                    "To improve:\n"
                    "  - If the observation has >15 features, try removing "
                    "the least essential ones\n"
                    "  - If the reward has >3 terms, try removing the weakest "
                    "ones to reduce noise\n"
                    "  - Make sure distance features are smooth and well-scaled"
                )
            elif sr < 0.9:
                hints.append(
                    "Good performance. Fine-tune:\n"
                    "  - Ensure the observation captures enough info for ALL "
                    "starting positions (corner cases)\n"
                    "  - Slightly increase the step penalty to push for "
                    "faster solutions\n"
                    "  - Check if the reward ever gives misleading signal "
                    "(e.g., facing bonus when agent should turn)"
                )
            else:
                hints.append(
                    "Excellent performance. To push higher:\n"
                    "  - Analyze which episodes still fail — are there edge "
                    "cases the observation doesn't capture?\n"
                    "  - Small tweaks to reward scale or step penalty may "
                    "help the last few percent"
                )

            # Plateau detection: parent ≈ best and best isn't great
            if best_sr > 0 and abs(sr - best_sr) < 0.1 and best_sr < 0.85:
                hints.append(
                    "**PLATEAU DETECTED**: The population has stagnated near "
                    f"{best_sr:.0%} success. Minor tweaks will not break "
                    "through. Try something STRUCTURALLY DIFFERENT:\n"
                    "  - If using many observation features, try drastically "
                    "fewer (e.g., just 6-8)\n"
                    "  - If using many reward terms, try just distance "
                    "shaping + goal bonus\n"
                    "  - Try Manhattan distance instead of Euclidean (or "
                    "vice versa)\n"
                    "  - Try a completely different observation encoding "
                    "(e.g., local grid patch instead of positions)\n"
                    "The key is to be BOLD — small changes won't escape "
                    "this plateau."
                )

            # Anti-patterns
            hints.append(
                "Avoid these common anti-patterns:\n"
                "  - Do NOT compute 'action values', 'turn scores', or "
                "'optimal direction' in get_observation — that is trying to "
                "solve the RL problem yourself and makes learning HARDER\n"
                "  - Do NOT use reward magnitudes >10 for any single term — "
                "this causes gradient instability\n"
                "  - Do NOT add features 'just in case' — every extra "
                "observation dimension makes learning slower"
            )

        if not hints:
            hints.append(
                "Design for learnability:\n"
                "- Keep the observation small (8-15 features) with raw, "
                "well-normalized information\n"
                "- Keep the reward simple: distance shaping + goal bonus "
                "+ step penalty\n"
                "- The neural network will learn the policy — your job is "
                "to give it good inputs and a clear signal"
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
