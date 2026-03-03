"""Brax PPO training for MuJoCo tasks in MDP Interface Discovery.

Uses brax.training.agents.ppo internally. Returns the same dict format
as train.py so the evaluator and controller are agnostic to the backend.
"""

import functools
import logging
import math
import time
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp

from mdp_discovery.adapters.base import EnvAdapter
from mdp_discovery.config import Config
from mdp_discovery.mdp_interface import MDPInterface

logger = logging.getLogger(__name__)


def _build_train_fn(config: Config, total_timesteps: int):
    """Build the Brax PPO train_fn partial (expensive — triggers XLA compilation).

    Separated so multi-seed runs can reuse the same compiled function.
    """
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground._src import wrapper

    tc = config.training
    policy_layers = tuple(tc.brax_policy_layers or [32, 32, 32, 32])
    value_layers = tuple(tc.brax_value_layers or [256, 256, 256, 256, 256])

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_layers,
        value_hidden_layer_sizes=value_layers,
        policy_obs_key="state",
        value_obs_key="state",
    )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=total_timesteps,
        num_evals=tc.brax_num_evals,
        reward_scaling=1.0,
        episode_length=tc.brax_episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=tc.brax_unroll_length,
        num_minibatches=tc.num_minibatches,
        num_updates_per_batch=tc.brax_updates_per_batch,
        discounting=tc.gamma,
        learning_rate=tc.lr,
        entropy_cost=tc.ent_coef,
        num_envs=tc.num_envs,
        batch_size=tc.brax_batch_size,
        max_grad_norm=tc.max_grad_norm,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    return train_fn


def _run_single_seed(
    train_fn,
    adapter: EnvAdapter,
    mdp_interface: MDPInterface,
    obs_dim: int,
    seed: int,
    episode_length: int = 500,
) -> Dict[str, Any]:
    """Run one seed using an already-built train_fn. Reuses XLA cache across seeds."""
    from mdp_discovery.adapters.mujoco_adapter import MjxMDPInterfaceWrapper

    get_obs_fn = mdp_interface.get_observation
    reward_fn = mdp_interface.compute_reward

    wrapped_env = MjxMDPInterfaceWrapper(
        adapter.make_task_env(), get_obs_fn, reward_fn, obs_dim=obs_dim
    )
    eval_env = MjxMDPInterfaceWrapper(
        adapter.make_task_env(), get_obs_fn, reward_fn, obs_dim=obs_dim
    )

    metrics_history: List[Dict[str, Any]] = []
    nan_detected = False
    t0 = time.time()

    def progress_fn(num_steps, metrics):
        nonlocal nan_detected
        reward = float(metrics.get("eval/episode_reward", 0.0))
        raw_success = float(metrics.get("eval/episode_success", 0.0))
        success = raw_success / max(episode_length, 1)
        elapsed = time.time() - t0

        # NaN/Inf detection — log warning but keep last good values
        if math.isnan(reward) or math.isinf(reward):
            if not nan_detected:
                logger.warning(
                    "  [%6.1fs] seed=%d NaN/Inf detected in reward at step %s — "
                    "using last good metrics from here on",
                    elapsed, seed, f"{num_steps:,}",
                )
                nan_detected = True
            return  # Don't record NaN entries

        if math.isnan(success) or math.isinf(success):
            success = 0.0  # Clamp bad success but keep reward

        metrics_history.append({
            "steps": num_steps,
            "reward": reward,
            "success": success,
        })
        logger.info(
            "  [%6.1fs] seed=%d step=%12s  reward=%.1f  success=%.2f",
            elapsed, seed, f"{num_steps:,}", reward, success,
        )

    make_inference_fn, params, _ = train_fn(
        environment=wrapped_env,
        progress_fn=progress_fn,
        eval_env=eval_env,
        seed=seed,
    )

    training_time = time.time() - t0

    if metrics_history:
        final = metrics_history[-1]
        final_return = final["reward"]
        final_success = final["success"]
    else:
        final_return = 0.0
        final_success = 0.0

    return {
        "final_return": final_return,
        "final_length": float(episode_length),
        "success_rate": final_success,
        "learning_curve": [m["reward"] for m in metrics_history],
        "success_curve": [m["success"] for m in metrics_history],
        "training_time": training_time,
        "nan_detected": nan_detected,
    }


def run_training(
    config: Config,
    adapter: EnvAdapter,
    mdp_interface: MDPInterface,
    obs_dim: int,
    total_timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Brax PPO training and return metrics.

    Args:
        config: Full system configuration.
        adapter: MuJoCo environment adapter.
        mdp_interface: Loaded MDP interface with get_observation and/or compute_reward.
        obs_dim: Observation dimension (detected by crash filter).
        total_timesteps: Override total timesteps (for cascade evaluation stages).

    Returns:
        Dictionary with: final_return, final_length, success_rate,
        learning_curve, success_curve, training_time.
    """
    tc = config.training
    if total_timesteps is None:
        total_timesteps = tc.total_timesteps

    logger.info(
        "Starting Brax PPO training: %s steps, %d envs, episode_length=%d, seed=%d",
        f"{total_timesteps:,}", tc.num_envs, tc.brax_episode_length, tc.seed,
    )

    # Build train_fn once — XLA compilation happens on first call, then cached
    train_fn = _build_train_fn(config, total_timesteps)
    return _run_single_seed(train_fn, adapter, mdp_interface, obs_dim, tc.seed, tc.brax_episode_length)


def run_training_multi_seed(
    config: Config,
    adapter: EnvAdapter,
    mdp_interface: MDPInterface,
    obs_dim: int,
    total_timesteps: Optional[int] = None,
    num_seeds: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Brax PPO training across multiple seeds WITHOUT recompiling.

    Builds train_fn once, then runs it N times with different seeds.
    XLA cache means seed 2+ skip compilation entirely.

    Returns averaged metrics across all seeds.
    """
    tc = config.training
    if total_timesteps is None:
        total_timesteps = tc.total_timesteps
    n = num_seeds or tc.num_seeds
    if n <= 1:
        return run_training(config, adapter, mdp_interface, obs_dim, total_timesteps)

    logger.info(
        "Starting Brax PPO multi-seed training: %s steps, %d seeds",
        f"{total_timesteps:,}", n,
    )

    # Build train_fn ONCE — all seeds share the same compiled program
    train_fn = _build_train_fn(config, total_timesteps)

    all_success = []
    all_return = []
    all_curves = []
    all_learning_curves = []
    total_time = 0.0
    any_nan = False

    for i in range(n):
        seed = tc.seed + i
        logger.info("  --- Seed %d/%d (seed=%d) ---", i + 1, n, seed)
        m = _run_single_seed(train_fn, adapter, mdp_interface, obs_dim, seed, tc.brax_episode_length)
        all_success.append(m["success_rate"])
        all_return.append(m["final_return"])
        total_time += m["training_time"]
        if m.get("nan_detected"):
            any_nan = True
        if m.get("success_curve"):
            all_curves.append(m["success_curve"])
        if m.get("learning_curve"):
            all_learning_curves.append(m["learning_curve"])
        logger.info(
            "  Seed %d/%d done: success=%.2f reward=%.1f time=%.1fs%s",
            i + 1, n, m["success_rate"], m["final_return"], m["training_time"],
            " (NaN detected)" if m.get("nan_detected") else "",
        )

    avg_success = sum(all_success) / n
    avg_return = sum(all_return) / n

    avg_curve = []
    if all_curves:
        min_len = min(len(c) for c in all_curves)
        avg_curve = [sum(c[j] for c in all_curves) / n for j in range(min_len)]

    avg_learning_curve = []
    if all_learning_curves:
        min_len = min(len(c) for c in all_learning_curves)
        avg_learning_curve = [sum(c[j] for c in all_learning_curves) / n for j in range(min_len)]

    # Per-seed success rates for stability analysis
    seed_spread = max(all_success) - min(all_success) if len(all_success) > 1 else 0.0

    logger.info(
        "  Multi-seed avg (%d seeds): success=%.2f reward=%.1f spread=%.2f",
        n, avg_success, avg_return, seed_spread,
    )

    return {
        "final_return": avg_return,
        "final_length": float(tc.brax_episode_length),
        "success_rate": avg_success,
        "success_curve": avg_curve,
        "learning_curve": avg_learning_curve,
        "training_time": total_time / n,  # Average, not sum
        "nan_detected": any_nan,
        "per_seed_success": all_success,
        "seed_spread": seed_spread,
    }
