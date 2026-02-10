"""Three-stage crash filter for candidate MDP interfaces.

Stage 0: Syntax check (ast.parse) + verify required functions exist.
Stage 1: Import check (importlib) — catches bad imports, JAX vs numpy issues.
Stage 2: Dry-run with real xland-minigrid State — catches shape mismatches,
         wrong attribute access, JAX tracing errors.
"""

import ast
import traceback
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp

import xminigrid

from mdp_discovery.config import Config
from mdp_discovery.mdp_interface import MDPInterface


@dataclass
class CrashFilterResult:
    """Result of running the crash filter on candidate code."""

    passed: bool
    stage_failed: Optional[int] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    obs_dim: Optional[int] = None


def _check_stage0(code: str) -> Optional[CrashFilterResult]:
    """Stage 0: Syntax check + verify required functions."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CrashFilterResult(
            passed=False,
            stage_failed=0,
            error_message=f"SyntaxError: {e}",
            error_traceback=traceback.format_exc(),
        )

    func_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = []
    if "get_observation" not in func_names:
        missing.append("get_observation")
    if "compute_reward" not in func_names:
        missing.append("compute_reward")

    if missing:
        return CrashFilterResult(
            passed=False,
            stage_failed=0,
            error_message=f"Missing required functions: {', '.join(missing)}",
        )
    return None


def _check_stage1(code: str) -> tuple[Optional[CrashFilterResult], Optional[MDPInterface]]:
    """Stage 1: Import check — load the module."""
    try:
        interface = MDPInterface.from_code(code)
    except Exception as e:
        return (
            CrashFilterResult(
                passed=False,
                stage_failed=1,
                error_message=f"{type(e).__name__}: {e}",
                error_traceback=traceback.format_exc(),
            ),
            None,
        )
    return None, interface


def _check_stage2(
    interface: MDPInterface, config: Config
) -> Optional[CrashFilterResult]:
    """Stage 2: Dry-run with a real xland-minigrid State."""
    try:
        env, env_params = xminigrid.make(config.environment.env_id)

        if config.environment.benchmark_id is not None and config.environment.ruleset_id is not None:
            benchmark = xminigrid.load_benchmark(config.environment.benchmark_id)
            env_params = env_params.replace(
                ruleset=benchmark.get_ruleset(config.environment.ruleset_id)
            )

        key = jax.random.key(0)
        timestep = env.reset(env_params, key)
        dummy_state = timestep.state

        obs_dim = interface.validate(dummy_state, max_obs_dim=config.mdp_interface.max_obs_dim)

        return None  # passed
    except Exception as e:
        return CrashFilterResult(
            passed=False,
            stage_failed=2,
            error_message=f"{type(e).__name__}: {e}",
            error_traceback=traceback.format_exc(),
        )


def run_crash_filter(code: str, config: Config) -> CrashFilterResult:
    """Run the full 3-stage crash filter on candidate MDP interface code.

    Returns a CrashFilterResult. If passed is True, obs_dim contains
    the detected observation dimension for the training pipeline.
    """
    # Stage 0
    result = _check_stage0(code)
    if result is not None:
        return result

    # Stage 1
    result, interface = _check_stage1(code)
    if result is not None:
        return result

    # Stage 2
    result = _check_stage2(interface, config)
    if result is not None:
        return result

    return CrashFilterResult(
        passed=True,
        obs_dim=interface.obs_dim,
    )
