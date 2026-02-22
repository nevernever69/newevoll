#!/usr/bin/env python3
"""Train best evolved interfaces on easy/medium/hard tasks and generate visualizations.

For each task:
  1. Trains the best interface for 5M timesteps (3 seeds, averaged)
  2. Renders a sample rollout GIF of the trained agent
  3. Renders a static image of the initial environment state

Usage:
    python benchmark_tasks.py
    python benchmark_tasks.py --timesteps 5000000 --seeds 3 --output-dir results/
"""

import argparse
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xminigrid
from xminigrid.rendering.rgb_render import render as rgb_render

from mdp_discovery.config import Config
from mdp_discovery.mdp_interface import MDPInterface
from mdp_discovery.train import make_states, run_training
from mdp_discovery.adapters.xminigrid_adapter import XMinigridAdapter


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

@dataclass
class TaskDef:
    name: str
    label: str  # Short label for plots
    config_path: str
    interface_path: str
    color: str  # Plot color
    task_description: str
    timesteps: int = 5_000_000  # Per-task training timesteps


TASKS = [
    TaskDef(
        name="easy_pickup",
        label="Easy: Pick Up",
        config_path="configs/easy_pickup.yaml",
        interface_path="runs/easy_pickup/best_interface.py",
        color="#4CAF50",
        task_description="Pick up the blue pyramid (9x9, single room)",
        timesteps=2_000_000,
    ),
    TaskDef(
        name="medium_place_near",
        label="Medium: Place Near",
        config_path="configs/medium_place_near.yaml",
        interface_path="runs/XLand-R1-9x9_20260221_064342/best_interface.py",
        color="#2196F3",
        task_description="Place blue pyramid adjacent to red square (9x9, single room)",
        timesteps=3_000_000,
    ),
    TaskDef(
        name="hard_rule_chain",
        label="Hard: Rule Chain",
        config_path="configs/hard_rule_chain.yaml",
        interface_path="runs/XLand-R4-13x13_20260221_074922/best_interface.py",
        color="#F44336",
        task_description="Pick up blue pyramid (transforms to green ball), place near yellow hex (13x13, 4 rooms)",
        timesteps=10_000_000,
    ),
]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def render_initial_state(config: Config, output_path: str):
    """Render the initial environment state as a PNG."""
    env, env_params = xminigrid.make(config.environment.env_id)

    if config.environment.ruleset_file is not None:
        spec = importlib.util.spec_from_file_location("_rs", config.environment.ruleset_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env_params = env_params.replace(ruleset=mod.build_ruleset())

    key = jax.random.key(42)
    timestep = env.reset(env_params, key)
    img = rgb_render(np.asarray(timestep.state.grid), timestep.state.agent, config.environment.view_size)
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  Initial state saved to {output_path}")


def render_rollout_gif(
    config: Config,
    interface: MDPInterface,
    output_path: str,
    max_steps: int = 200,
    fps: int = 4,
):
    """Run a single episode with the trained-ish interface and save as GIF."""
    env, env_params = xminigrid.make(config.environment.env_id)

    if config.environment.ruleset_file is not None:
        spec = importlib.util.spec_from_file_location("_rs", config.environment.ruleset_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env_params = env_params.replace(ruleset=mod.build_ruleset())

    env_params = env_params.replace(max_steps=max_steps)
    view_size = config.environment.view_size

    key = jax.random.key(123)
    timestep = env.reset(env_params, key)

    frames = []
    frames.append(rgb_render(np.asarray(timestep.state.grid), timestep.state.agent, view_size))

    for step in range(max_steps):
        key, action_key = jax.random.split(key)
        # Random policy for visualization (just shows the environment)
        action = jax.random.randint(action_key, (), 0, 6)
        timestep = env.step(env_params, timestep, action)
        frames.append(rgb_render(np.asarray(timestep.state.grid), timestep.state.agent, view_size))

        # Stop if episode ended
        if timestep.step_type == 2:  # LAST
            break

    imageio.mimsave(output_path, frames, duration=1000 // fps, loop=0)
    print(f"  Rollout GIF saved to {output_path} ({len(frames)} frames)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_interface(
    task: TaskDef,
    num_seeds: int,
) -> Dict:
    """Train a task's interface for task-specific timesteps across seeds, return metrics."""
    total_timesteps = task.timesteps
    config = Config.from_yaml(task.config_path)
    interface = MDPInterface.from_file(task.interface_path)
    adapter = XMinigridAdapter(config)
    obs_dim = interface.detect_obs_dim(adapter.get_dummy_state())

    print(f"\n  Interface: {task.interface_path}")
    print(f"  Obs dim: {obs_dim}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Seeds: {num_seeds}")

    seed_results = []
    for seed in range(num_seeds):
        config_copy = Config.from_yaml(task.config_path)
        config_copy.training.seed = 42 + seed

        print(f"  [seed={seed}] training ... ", end="", flush=True)
        t0 = time.time()
        metrics = run_training(
            config_copy,
            XMinigridAdapter(config_copy),
            interface,
            obs_dim,
            total_timesteps=total_timesteps,
        )
        elapsed = time.time() - t0
        print(f"success={metrics['success_rate']*100:.1f}%  length={metrics['final_length']:.1f}  ({elapsed:.1f}s)")

        seed_results.append({
            "seed": seed,
            "success_rate": float(metrics["success_rate"]),
            "final_return": float(metrics["final_return"]),
            "final_length": float(metrics["final_length"]),
            "training_time": float(metrics["training_time"]),
            "success_curve": [float(x) for x in metrics["success_curve"]],
        })

    # Compute summary
    rates = [r["success_rate"] for r in seed_results]
    lengths = [r["final_length"] for r in seed_results]
    times = [r["training_time"] for r in seed_results]

    return {
        "task": task.name,
        "total_timesteps": total_timesteps,
        "obs_dim": obs_dim,
        "seeds": seed_results,
        "mean_success": float(np.mean(rates)),
        "std_success": float(np.std(rates)),
        "mean_length": float(np.mean(lengths)),
        "mean_time": float(np.mean(times)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(all_results: Dict[str, Dict], output_dir: str):
    """Generate comparison plots."""

    # --- Figure 1: Bar chart comparing final success rates ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Evolved MDP Interfaces — Final Performance (5M Steps)", fontsize=14, fontweight="bold")

    names = []
    means = []
    stds = []
    colors = []
    descriptions = []

    for task in TASKS:
        if task.name in all_results:
            r = all_results[task.name]
            names.append(task.label)
            means.append(r["mean_success"] * 100)
            stds.append(r["std_success"] * 100)
            colors.append(task.color)
            descriptions.append(task.task_description)

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.5, capsize=10,
                  edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, val, desc in zip(bars, means, descriptions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)
        ax.text(bar.get_x() + bar.get_width() / 2, -8,
                desc, ha="center", va="top", fontsize=7, style="italic", color="gray",
                wrap=True)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "task_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved to {path}")

    # --- Figure 2: Learning curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Learning Curves — Success Rate During Training", fontsize=14, fontweight="bold")

    for task in TASKS:
        if task.name not in all_results:
            continue
        r = all_results[task.name]
        seeds = r["seeds"]

        # Align curves (they may have different lengths)
        max_len = max(len(s["success_curve"]) for s in seeds)
        curves = np.full((len(seeds), max_len), np.nan)
        for i, s in enumerate(seeds):
            c = s["success_curve"]
            curves[i, :len(c)] = c

        mean_curve = np.nanmean(curves, axis=0) * 100
        std_curve = np.nanstd(curves, axis=0) * 100

        total_ts = r["total_timesteps"]
        xs = np.linspace(0, total_ts / 1_000_000, len(mean_curve))

        ax.plot(xs, mean_curve, color=task.color, label=task.label, linewidth=2)
        ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve,
                        color=task.color, alpha=0.15)

    ax.set_xlabel("Training Steps (millions)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Learning curves saved to {path}")


def render_env_overview(output_dir: str):
    """Render side-by-side initial states of all 3 environments."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Environment Overview", fontsize=16, fontweight="bold", y=1.02)

    for idx, task in enumerate(TASKS):
        config = Config.from_yaml(task.config_path)
        env, env_params = xminigrid.make(config.environment.env_id)

        if config.environment.ruleset_file is not None:
            spec = importlib.util.spec_from_file_location("_rs", config.environment.ruleset_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            env_params = env_params.replace(ruleset=mod.build_ruleset())

        key = jax.random.key(42)
        timestep = env.reset(env_params, key)
        img = rgb_render(np.asarray(timestep.state.grid), timestep.state.agent, config.environment.view_size)

        axes[idx].imshow(img)
        axes[idx].set_title(f"{task.label}\n{task.task_description}", fontsize=10, pad=10)
        axes[idx].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "environment_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Environment overview saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark evolved interfaces on easy/medium/hard tasks")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Training timesteps per task")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per task")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only generate visualizations")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task names to run (default: all)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Filter tasks if specified
    tasks = TASKS
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
        tasks = [t for t in TASKS if t.name in task_names]
        if not tasks:
            print(f"ERROR: No matching tasks found. Available: {[t.name for t in TASKS]}")
            return

    print("=" * 70)
    print("MDP Interface Benchmark — Easy / Medium / Hard")
    print("=" * 70)
    print(f"  Tasks:      {[t.name for t in tasks]}")
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Output:     {output_dir}/")
    print(f"  Devices:    {jax.local_device_count()}")
    print("=" * 70)

    # --- Step 1: Render environment visualizations ---
    print("\n--- Rendering environment visualizations ---")
    render_env_overview(output_dir)

    for task in tasks:
        config = Config.from_yaml(task.config_path)
        task_dir = os.path.join(output_dir, task.name)
        os.makedirs(task_dir, exist_ok=True)

        render_initial_state(config, os.path.join(task_dir, "initial_state.png"))
        max_steps = config.training.eval_max_steps
        render_rollout_gif(config, None, os.path.join(task_dir, "random_rollout.gif"),
                           max_steps=min(max_steps, 100), fps=4)

    # --- Step 2: Train all tasks ---
    all_results = {}

    if not args.skip_training:
        print("\n--- Training interfaces ---")
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Task: {task.label} — {task.task_description}")
            print(f"{'='*60}")

            result = train_interface(task, args.seeds)
            all_results[task.name] = result

            print(f"\n  Summary: {result['mean_success']*100:.1f}% +/- {result['std_success']*100:.1f}%")
            print(f"  Mean episode length: {result['mean_length']:.1f}")
            print(f"  Mean training time: {result['mean_time']:.1f}s")

        # Save raw results
        json_path = os.path.join(output_dir, "benchmark_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nRaw results saved to {json_path}")

        # --- Step 3: Generate plots ---
        print("\n--- Generating plots ---")
        plot_results(all_results, output_dir)
    else:
        # Try to load existing results
        json_path = os.path.join(output_dir, "benchmark_results.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                all_results = json.load(f)
            print(f"\nLoaded existing results from {json_path}")
            plot_results(all_results, output_dir)
        else:
            print("\nNo existing results found. Run without --skip-training first.")

    # --- Print summary table ---
    if all_results:
        print(f"\n{'='*70}")
        print(f"{'Task':<30} {'Success':>12} {'Length':>10} {'Time':>10}")
        print(f"{'='*70}")
        for task in tasks:
            if task.name in all_results:
                r = all_results[task.name]
                print(f"{task.label:<30} {r['mean_success']*100:>8.1f}% +/- {r['std_success']*100:.1f}%"
                      f" {r['mean_length']:>8.1f}  {r['mean_time']:>8.1f}s")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
