#!/usr/bin/env python3
"""CLI entry point for MDP Interface Discovery.

Usage:
    # Basic run with defaults (MiniGrid-Empty-6x6, claude-4-sonnet)
    python run.py --task "Navigate to the goal tile as quickly as possible."

    # Custom config
    python run.py --config configs/default.yaml \
                  --task "Pick up the key, unlock the door, and reach the goal." \
                  --model claude-4-sonnet \
                  --iterations 100 \
                  --checkpoint-dir runs/doorkey_01

    # Resume from checkpoint
    python run.py --task "Navigate to the goal tile." \
                  --resume runs/doorkey_01

    # Override environment
    python run.py --task "Navigate to the goal in a large room." \
                  --env MiniGrid-Empty-8x8 \
                  --iterations 50

    # Ablation modes
    python run.py --task "Navigate to the goal." --mode reward_only
    python run.py --task "Navigate to the goal." --mode default --iterations 1
    python run.py --task "Navigate to the goal." --mode random --iterations 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from mdp_discovery.config import Config
from mdp_discovery.controller import EvolutionController


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDP Interface Discovery via LLM-Guided Evolutionary Search",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Natural language task description for the RL agent. "
             "If not specified, uses config's task_description.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for checkpoints. Auto-generated if not specified.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint directory.",
    )

    # Mode and context overrides
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "reward_only", "obs_only", "default", "random"],
        default=None,
        help="Evolution mode override (default: from config).",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Path to context file override.",
    )

    # Common overrides
    parser.add_argument("--env", type=str, default=None, help="Environment ID override.")
    parser.add_argument("--model", type=str, default=None, help="LLM model name override.")
    parser.add_argument("--region", type=str, default=None, help="AWS region override.")
    parser.add_argument("--iterations", type=int, default=None, help="Max iterations override.")
    parser.add_argument("--timesteps", type=int, default=None, help="Short training timesteps override.")
    parser.add_argument("--timesteps-full", type=int, default=None, help="Full training timesteps override.")
    parser.add_argument("--cascade-threshold", type=float, default=None, help="Cascade threshold override.")
    parser.add_argument("--candidates", type=int, default=None, help="Number of candidates to evaluate in parallel per batch.")
    parser.add_argument("--no-cascade", action="store_true", help="Disable cascade (go straight to full training).")
    parser.add_argument("--log-level", type=str, default=None, help="Logging level (DEBUG, INFO, WARNING).")

    return parser.parse_args()


def setup_logging(log_level: str, log_dir: str):
    """Configure logging: detailed logs to file, clean progress to console."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "evolution.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler: everything, detailed format
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler: only INFO+, compact format
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    # Suppress noisy libraries on console
    for name in ("botocore", "boto3", "urllib3", "jax", "flax"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return str(log_file)


def main():
    args = parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Apply overrides
    if args.env:
        config.environment.env_id = args.env
    if args.model:
        config.llm.model_name = args.model
    if args.region:
        config.llm.region_name = args.region
    if args.iterations:
        config.max_iterations = args.iterations
    if args.timesteps:
        config.training.total_timesteps = args.timesteps
    if args.timesteps_full:
        config.training.total_timesteps_full = args.timesteps_full
    if args.cascade_threshold is not None:
        config.evaluator.cascade_thresholds = [args.cascade_threshold]
    if args.candidates:
        config.candidates_per_iteration = args.candidates
    if args.no_cascade:
        config.evaluator.cascade_evaluation = False
    if args.log_level:
        config.log_level = args.log_level
    if args.mode:
        config.evolution_mode = args.mode
    if args.context:
        config.environment.context_file = args.context

    # Resolve task: CLI arg overrides config
    task_description = args.task  # may be None; controller falls back to config

    # Auto-generate checkpoint dir if not specified
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and args.resume is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        env_short = config.environment.env_id.replace("MiniGrid-", "").replace("XLand-MiniGrid-", "")
        mode_suffix = f"_{config.evolution_mode}" if config.evolution_mode != "full" else ""
        checkpoint_dir = f"runs/{env_short}{mode_suffix}_{timestamp}"

    run_dir = checkpoint_dir or args.resume

    # Setup logging (file + clean console)
    log_file = setup_logging(config.log_level, run_dir)

    # Create controller
    controller = EvolutionController(
        config=config,
        task_description=task_description,
        checkpoint_dir=run_dir,
    )

    # Resume if specified
    if args.resume:
        controller.load_checkpoint(args.resume)

    # Print startup banner to console
    print(f"{'=' * 60}")
    print(f"MDP Interface Discovery")
    print(f"{'=' * 60}")
    print(f"  Task:        {controller.task_description}")
    print(f"  Environment: {config.environment.env_id}")
    print(f"  Mode:        {config.evolution_mode}")
    print(f"  Model:       {config.llm.model_name} ({config.llm.region_name})")
    print(f"  Iterations:  {config.max_iterations}")
    print(f"  Candidates:  {config.candidates_per_iteration} per batch")
    print(f"  Training:    {config.training.total_timesteps:,} short / {config.training.total_timesteps_full:,} full steps")
    print(f"  Cascade:     {'ON' if config.evaluator.cascade_evaluation else 'OFF'} (thresholds={config.evaluator.cascade_thresholds})")
    print(f"  Context:     {config.environment.context_file}")
    print(f"  Run dir:     {run_dir}")
    print(f"  Log file:    {log_file}")
    print(f"{'=' * 60}")
    print()

    # Run
    controller.run()

    # Print final results
    best = controller.db.get_best_program()
    print()
    print(f"{'=' * 60}")
    print(f"Evolution complete — {controller.iteration} iterations in {controller.total_eval_time:.0f}s")
    print(f"{'=' * 60}")
    if best:
        best_file = Path(run_dir) / "best_interface.py"
        print(f"  Best success:  {best.fitness:.0%}")
        print(f"  Best obs_dim:  {best.obs_dim}")
        print(f"  Best program:  {best_file}")
    else:
        print("  No successful programs found.")
    print(f"  Full logs:     {log_file}")
    print(f"  Checkpoint:    {run_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
