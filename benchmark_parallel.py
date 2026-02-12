#!/usr/bin/env python3
"""Benchmark parallel candidates: sequential vs 2/3/4 parallel workers.

Runs 4 iterations for each configuration, measures wall time per batch
and total wall time. Plots results with matplotlib.

Usage:
    python benchmark_parallel.py
    python benchmark_parallel.py --iterations 8 --timesteps 131072
"""

import argparse
import subprocess
import re
import time
import json
import tempfile
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_benchmark(candidates: int, iterations: int, timesteps: int, run_dir: str) -> dict:
    """Run one benchmark configuration and parse results."""
    cmd = [
        "python3", "run.py",
        "--task", "Navigate to the goal tile in a 6x6 empty room.",
        "--iterations", str(iterations),
        "--candidates", str(candidates),
        "--timesteps", str(timesteps),
        "--timesteps-full", str(timesteps),
        "--cascade-threshold", "0.0",
        "--checkpoint-dir", run_dir,
    ]

    print(f"\n{'='*60}")
    print(f"Running: candidates={candidates}, iterations={iterations}")
    print(f"{'='*60}")

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall_time = time.time() - start

    output = proc.stdout + proc.stderr

    # Parse wall time from the evolution summary
    m = re.search(r"Wall time:\s+([\d.]+)s", output)
    reported_wall = float(m.group(1)) if m else wall_time

    # Parse per-iteration times
    iter_times = re.findall(r"\[Iter \d+\].*?([\d.]+)s", output)
    iter_times = [float(t) for t in iter_times]

    # Parse batch submissions
    batch_lines = re.findall(r"Submitting batch of (\d+) candidate", output)
    num_batches = len(batch_lines)

    # Parse success rates
    successes = re.findall(r"success=([\d.]+)%", output)
    successes = [float(s) for s in successes]

    # Parse total eval time
    m = re.search(r"Total eval time:\s+([\d.]+)s", output)
    total_eval = float(m.group(1)) if m else sum(iter_times)

    result = {
        "candidates": candidates,
        "iterations": iterations,
        "wall_time": reported_wall,
        "measured_wall_time": wall_time,
        "num_batches": num_batches,
        "iter_times": iter_times,
        "success_rates": successes,
        "total_eval_time": total_eval,
        "time_per_iteration": reported_wall / iterations if iterations > 0 else 0,
    }

    print(f"  Wall time:       {reported_wall:.1f}s")
    print(f"  Batches:         {num_batches}")
    print(f"  Time/iteration:  {result['time_per_iteration']:.1f}s")
    print(f"  Total eval time: {total_eval:.1f}s")
    if successes:
        print(f"  Success rates:   {successes}")

    return result


def plot_results(results: list, output_path: str = "benchmark_parallel.png"):
    """Plot benchmark results."""
    candidates = [r["candidates"] for r in results]
    wall_times = [r["wall_time"] for r in results]
    time_per_iter = [r["time_per_iteration"] for r in results]
    total_eval = [r["total_eval_time"] for r in results]
    iterations = results[0]["iterations"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Parallel Candidates Benchmark ({iterations} iterations, "
        f"{results[0].get('timesteps', '?')} timesteps)",
        fontsize=14, fontweight="bold",
    )

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    # 1. Total wall time
    ax = axes[0]
    bars = ax.bar(candidates, wall_times, color=colors[:len(candidates)], width=0.6)
    ax.set_xlabel("Parallel Candidates")
    ax.set_ylabel("Wall Time (s)")
    ax.set_title("Total Wall Time")
    ax.set_xticks(candidates)
    for bar, val in zip(bars, wall_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}s", ha="center", va="bottom", fontweight="bold")

    # Add ideal scaling line
    if wall_times[0] > 0:
        ideal = [wall_times[0] / c for c in candidates]
        ax.plot(candidates, ideal, "k--", alpha=0.4, label="Ideal scaling")
        ax.legend()

    # 2. Time per iteration (amortized)
    ax = axes[1]
    bars = ax.bar(candidates, time_per_iter, color=colors[:len(candidates)], width=0.6)
    ax.set_xlabel("Parallel Candidates")
    ax.set_ylabel("Time per Iteration (s)")
    ax.set_title("Amortized Time per Iteration")
    ax.set_xticks(candidates)
    for bar, val in zip(bars, time_per_iter):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontweight="bold")

    # 3. Speedup relative to sequential
    ax = axes[2]
    if wall_times[0] > 0:
        speedups = [wall_times[0] / w for w in wall_times]
        bars = ax.bar(candidates, speedups, color=colors[:len(candidates)], width=0.6)
        ax.plot(candidates, candidates, "k--", alpha=0.4, label="Ideal (linear)")
        ax.set_xlabel("Parallel Candidates")
        ax.set_ylabel("Speedup (x)")
        ax.set_title("Speedup vs Sequential")
        ax.set_xticks(candidates)
        ax.legend()
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}x", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel candidates")
    parser.add_argument("--iterations", type=int, default=4, help="Iterations per run")
    parser.add_argument("--timesteps", type=int, default=131072, help="Training timesteps (short + full)")
    parser.add_argument("--max-candidates", type=int, default=4, help="Max parallel candidates to test")
    parser.add_argument("--output", type=str, default="benchmark_parallel.png", help="Output plot path")
    args = parser.parse_args()

    candidate_counts = list(range(1, args.max_candidates + 1))
    results = []

    for c in candidate_counts:
        run_dir = tempfile.mkdtemp(prefix=f"bench_c{c}_")
        try:
            r = run_benchmark(c, args.iterations, args.timesteps, run_dir)
            r["timesteps"] = args.timesteps
            results.append(r)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT for candidates={c}, skipping")
        except Exception as e:
            print(f"  ERROR for candidates={c}: {e}")
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    if len(results) < 2:
        print("Not enough results to plot.")
        return

    # Save raw data
    json_path = args.output.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw data saved to {json_path}")

    plot_results(results, args.output)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Candidates':>12} {'Wall Time':>12} {'Time/Iter':>12} {'Speedup':>10}")
    print(f"{'='*60}")
    base = results[0]["wall_time"]
    for r in results:
        speedup = base / r["wall_time"] if r["wall_time"] > 0 else 0
        print(f"{r['candidates']:>12d} {r['wall_time']:>11.1f}s {r['time_per_iteration']:>11.1f}s {speedup:>9.2f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
