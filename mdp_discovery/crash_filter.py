"""Three-stage crash filter for candidate MDP interfaces.

Stage 0: Syntax check (ast.parse) + verify required functions exist.
Stage 1: Import check (importlib) — catches bad imports, JAX vs numpy issues.
Stage 2: Dry-run with a real environment State — catches shape mismatches,
         wrong attribute access, JAX tracing errors.
"""

import ast
import traceback
from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp

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


def _check_stage0(code: str, required_functions: List[str]) -> Optional[CrashFilterResult]:
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
    missing = [fn for fn in required_functions if fn not in func_names]

    if missing:
        return CrashFilterResult(
            passed=False,
            stage_failed=0,
            error_message=f"Missing required functions: {', '.join(missing)}",
        )
    return None


def _check_stage1(
    code: str, required_functions: List[str]
) -> tuple[Optional[CrashFilterResult], Optional[MDPInterface]]:
    """Stage 1: Import check — load the module."""
    try:
        interface = MDPInterface.from_code(code, required_functions=required_functions)
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
    interface: MDPInterface, config: Config, dummy_state
) -> Optional[CrashFilterResult]:
    """Stage 2: Dry-run with a real environment State."""
    try:
        obs_dim = interface.validate(
            dummy_state,
            max_obs_dim=config.mdp_interface.max_obs_dim,
        )
        return None  # passed
    except Exception as e:
        return CrashFilterResult(
            passed=False,
            stage_failed=2,
            error_message=f"{type(e).__name__}: {e}",
            error_traceback=traceback.format_exc(),
        )


def run_crash_filter(
    code: str,
    config: Config,
    dummy_state,
    required_functions: Optional[List[str]] = None,
) -> CrashFilterResult:
    """Run the full 3-stage crash filter on candidate MDP interface code.

    Args:
        code: Python source code to validate.
        config: System configuration.
        dummy_state: A real environment state for dry-run validation.
        required_functions: List of function names to check for.
            Defaults to ["get_observation", "compute_reward"].

    Returns:
        A CrashFilterResult. If passed is True, obs_dim contains
        the detected observation dimension for the training pipeline.
    """
    if required_functions is None:
        required_functions = ["get_observation", "compute_reward"]

    # Stage 0
    result = _check_stage0(code, required_functions)
    if result is not None:
        return result

    # Stage 1
    result, interface = _check_stage1(code, required_functions)
    if result is not None:
        return result

    # Stage 2
    result = _check_stage2(interface, config, dummy_state)
    if result is not None:
        return result

    return CrashFilterResult(
        passed=True,
        obs_dim=interface.obs_dim,
    )
