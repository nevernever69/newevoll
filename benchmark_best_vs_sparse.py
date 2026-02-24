#!/usr/bin/env python3
"""Benchmark: Best Evolved Interface vs Ablation Spaces vs Sparse Baseline.

Per-task budgets: Easy 2M, Medium 4M, Hard 6M.
Runs with 10 seeds per method for reduced variance.
- evolved: full MDP interface (obs + reward) from main experiments
- obs_only: ablation observation + env built-in reward
- reward_only: ablation reward + default observation  
- sparse: adapter.get_default_obs_fn() + env built-in reward
"""

import json
import os
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mdp_discovery.config import Config
from mdp_discovery.mdp_interface import MDPInterface
from mdp_discovery.train import run_training
from mdp_discovery.adapters.xminigrid_adapter import XMinigridAdapter

BASE_SEED = 42
NUM_SEEDS = 10

TASKS = [
    {
        "name": "Easy\n(Pick up blue pyramid)",
        "short": "easy",
        "config": "configs/easy_pickup.yaml",
        "evolved": "experiments/main/easy/best_interface.py",
        "obs_only": "experiments/ablations/easy_obs_only/best_interface.py",
        "reward_only": "experiments/ablations/easy_reward_only/best_interface.py",
        "timesteps": 2_000_000,
    },
    {
        "name": "Medium\n(Place pyramid near square)",
        "short": "medium",
        "config": "configs/medium_place_near.yaml",
        "evolved": "experiments/main/medium/best_interface.py",
        "obs_only": "experiments/ablations/medium_obs_only/best_interface.py",
        "reward_only": "experiments/ablations/medium_reward_only/best_interface.py",
        "timesteps": 4_000_000,
    },
    {
        "name": "Hard\n(Rule chain + placement)",
        "short": "hard",
        "config": "configs/hard_rule_chain.yaml",
        "evolved": "experiments/main/hard/best_interface.py",
        "obs_only": "experiments/ablations/hard_obs_only/best_interface.py",
        "reward_only": "experiments/ablations/hard_reward_only/best_interface.py",
        "timesteps": 6_000_000,
    },
]


def main():
    print("=" * 70)
    print("Best Evolved Interface vs Ablation Spaces vs Sparse Baseline")
    print(f"  Seeds: {NUM_SEEDS} (base={BASE_SEED}) | Devices: {jax.local_device_count()}")
    print(f"  Budgets: Easy=2M, Medium=4M, Hard=6M")
    print(f"  obs_only = ablation obs + env built-in reward")
    print(f"  reward_only = default obs + ablation reward")
    print(f"  sparse = default obs + env built-in reward")
    print("=" * 70)

    all_results = {}

    for task in TASKS:
        ts = task["timesteps"]
        print(f"\n{'─'*70}")
        print(f"Task: {task['short'].upper()} — {ts/1e6:.0f}M steps × {NUM_SEEDS} seeds")
        print(f"{'─'*70}")

        config = Config.from_yaml(task["config"])
        config.training.lr_schedule = "cosine"
        adapter = XMinigridAdapter(config)
        dummy_state = adapter.get_dummy_state()

        # Load interfaces once per task
        evolved = MDPInterface.from_file(task["evolved"])
        evolved_obs = evolved.validate(dummy_state)
        print(f"  Evolved obs_dim: {evolved_obs}")

        obs_only_interface = MDPInterface.from_file(task["obs_only"], required_functions=["get_observation"])
        obs_only_interface.validate(dummy_state)
        obs_only_obs = obs_only_interface.obs_dim
        print(f"  obs_only obs_dim:  {obs_only_obs}")

        reward_only_interface = MDPInterface.from_file(task["reward_only"], required_functions=["compute_reward"])
        reward_only_interface.validate(dummy_state)
        default_obs_fn = adapter.get_default_obs_fn()
        test_obs = default_obs_fn(dummy_state)
        reward_only_obs = int(jnp.asarray(test_obs).shape[0])
        print(f"  reward_only obs_dim: {reward_only_obs}")

        sparse_obs = int(jnp.asarray(adapter.get_default_obs_fn()(dummy_state)).shape[0])
        print(f"  Sparse obs_dim:  {sparse_obs}")

        # Run all seeds for all methods
        task_results = {}
        for label, interface, obs_dim in [
            ("evolved", evolved, evolved_obs),
            ("obs_only", MDPInterface(get_observation=obs_only_interface.get_observation, compute_reward=adapter.get_default_reward_fn()), obs_only_obs),
            ("reward_only", MDPInterface(get_observation=default_obs_fn, compute_reward=reward_only_interface.compute_reward), reward_only_obs),
            ("sparse", MDPInterface(get_observation=adapter.get_default_obs_fn(), compute_reward=None), sparse_obs),
        ]:
            print(f"\n  Training {label} (obs_dim={obs_dim}) for {ts/1e6:.0f}M × {NUM_SEEDS} seeds...", flush=True)
            
            all_metrics = []
            all_curves = []
            seed_times = []
            
            for seed_idx in range(NUM_SEEDS):
                seed = BASE_SEED + seed_idx
                config.training.seed = seed
                print(f"    Seed {seed_idx+1}/{NUM_SEEDS} (seed={seed})...", flush=True)
                t0 = time.time()
                metrics = run_training(config, adapter, interface, obs_dim, total_timesteps=ts)
                elapsed = time.time() - t0
                seed_times.append(elapsed)
                all_metrics.append(metrics)
                all_curves.append(metrics["success_curve"])
                print(f"      {elapsed:.1f}s — success={metrics['success_rate']*100:.1f}%  len={metrics['final_length']:.1f}")
            
            # Average across seeds
            avg_success = np.mean([m["success_rate"] for m in all_metrics])
            avg_return = np.mean([m["final_return"] for m in all_metrics])
            avg_length = np.mean([m["final_length"] for m in all_metrics])
            avg_time = np.mean(seed_times)
            std_success = np.std([m["success_rate"] for m in all_metrics])
            
            # Average success curve
            avg_curve = np.mean(all_curves, axis=0).tolist()
            std_curve = np.std(all_curves, axis=0).tolist()
            
            print(f"    AVG: success={avg_success*100:.1f}%±{std_success*100:.1f}%  len={avg_length:.1f}  time={avg_time:.1f}s")
            
            task_results[label] = {
                "success_rate": avg_success,
                "std_success": std_success,
                "final_return": avg_return,
                "final_length": avg_length,
                "training_time": avg_time,
                "success_curve": avg_curve,
                "std_curve": std_curve,
                "all_curves": all_curves,  # Keep for plotting
            }

        all_results[task["short"]] = {
            "name": task["name"],
            "results": task_results,
            "timesteps": ts,
            "num_envs": config.training.num_envs,
            "num_steps": config.training.num_steps,
        }

    # ─── Save JSON ────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    json_path = "results/evolved_vs_sparse.json"
    json_data = {}
    for key, val in all_results.items():
        json_data[key] = {
            "name": val["name"],
            "timesteps": val["timesteps"],
            "num_envs": val["num_envs"],
            "num_steps": val["num_steps"],
        }
        for variant in ["evolved", "obs_only", "reward_only", "sparse"]:
            m = val["results"][variant]
            json_data[key][variant] = {
                "success_rate": m["success_rate"],
                "final_return": m["final_return"],
                "final_length": m["final_length"],
                "training_time": m["training_time"],
                "success_curve": m["success_curve"],
            }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ─── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    fig.suptitle(
        f"Evolved vs Ablation Spaces vs Sparse ({NUM_SEEDS} seeds, mean ± std)",
        fontsize=15, fontweight="bold", y=1.02,
    )

    palette = {
        "evolved": "#1976D2",      # blue
        "obs_only": "#388E3C",     # green
        "reward_only": "#F57C00",  # orange
        "sparse": "#D32F2F",       # red
    }
    nice_label = {
        "evolved": "Evolved (full obs + reward)",
        "obs_only": "obs_only (ablation obs + env reward)",
        "reward_only": "reward_only (default obs + ablation reward)",
        "sparse": "Sparse (default obs + env reward)",
    }

    for ax, task_key in zip(axes, ["easy", "medium", "hard"]):
        r = all_results[task_key]
        ts_per_update = r["num_envs"] * r["num_steps"]
        total_ts = r["timesteps"]

        for variant in ["evolved", "obs_only", "reward_only", "sparse"]:
            curve = r["results"][variant]["success_curve"]
            std_curve = r["results"][variant]["std_curve"]
            xs = [(i + 1) * ts_per_update / 1e6 for i in range(len(curve))]
            ys = [v * 100 for v in curve]
            std_ys = [s * 100 for s in std_curve]
            
            ax.plot(
                xs, ys,
                color=palette[variant],
                label=nice_label[variant],
                linewidth=2.2,
                alpha=0.9,
            )
            ax.fill_between(
                xs,
                [max(0, y - s) for y, s in zip(ys, std_ys)],
                [min(100, y + s) for y, s in zip(ys, std_ys)],
                color=palette[variant],
                alpha=0.2,
            )

            # Annotate final success rate with std
            final_sr = r["results"][variant]["success_rate"] * 100
            std_sr = r["results"][variant]["std_success"] * 100
            ax.plot(xs[-1], ys[-1], "o", color=palette[variant], markersize=7)
            
            # Adjust annotation offset to avoid overlap
            offsets = {
                ("easy", "evolved"): (-25, 6),
                ("easy", "obs_only"): (-25, -12),
                ("easy", "reward_only"): (-25, -28),
                ("easy", "sparse"): (-25, -44),
                ("medium", "evolved"): (-25, 6),
                ("medium", "obs_only"): (-25, -12),
                ("medium", "reward_only"): (-25, -28),
                ("medium", "sparse"): (-25, -44),
                ("hard", "evolved"): (-25, 6),
                ("hard", "obs_only"): (-25, -12),
                ("hard", "reward_only"): (-25, -28),
                ("hard", "sparse"): (-25, -44),
            }
            offset_x, offset_y = offsets.get((task_key, variant), (-25, 6))
            ax.annotate(
                f"{final_sr:.0f}%±{std_sr:.0f}%",
                xy=(xs[-1], ys[-1]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=9, fontweight="bold",
                color=palette[variant],
            )

        ax.set_title(r["name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Timesteps (millions)", fontsize=11)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.set_ylim(-5, 105)
        ax.set_xlim(0, total_ts / 1e6 + 0.1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out_path = "results/evolved_vs_sparse.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close()

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Task':<12} {'Budget':>8} {'Evolved':>14} {'obs_only':>14} {'rew_only':>14} {'Sparse':>14}")
    print(f"{'='*70}")
    for k in ["easy", "medium", "hard"]:
        budget = all_results[k]["timesteps"]
        e = all_results[k]["results"]["evolved"]["success_rate"] * 100
        e_std = all_results[k]["results"]["evolved"]["std_success"] * 100
        o = all_results[k]["results"]["obs_only"]["success_rate"] * 100
        o_std = all_results[k]["results"]["obs_only"]["std_success"] * 100
        r = all_results[k]["results"]["reward_only"]["success_rate"] * 100
        r_std = all_results[k]["results"]["reward_only"]["std_success"] * 100
        s = all_results[k]["results"]["sparse"]["success_rate"] * 100
        s_std = all_results[k]["results"]["sparse"]["std_success"] * 100
        print(f"{k:<12} {budget/1e6:>7.0f}M {e:>6.1f}%±{e_std:>4.1f} {o:>6.1f}%±{o_std:>4.1f} {r:>6.1f}%±{r_std:>4.1f} {s:>6.1f}%±{s_std:>4.1f}")
    print(f"{'='*70}")
    
    # ─── Delta Analysis ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Delta Analysis (vs evolved)")
    print(f"{'='*70}")
    for k in ["easy", "medium", "hard"]:
        e = all_results[k]["results"]["evolved"]["success_rate"] * 100
        o = all_results[k]["results"]["obs_only"]["success_rate"] * 100
        r = all_results[k]["results"]["reward_only"]["success_rate"] * 100
        s = all_results[k]["results"]["sparse"]["success_rate"] * 100
        print(f"{k}: obs_only={o-e:+.1f}%  reward_only={r-e:+.1f}%  sparse={s-e:+.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
