#!/usr/bin/env python3
"""Generate clean plots from ablation spaces benchmark results.

Reads: results/ablation_spaces.json
Outputs: 
  - results/ablation_spaces_combined.png (3-panel figure)
  - results/ablation_spaces_easy.png (individual)
  - results/ablation_spaces_medium.png (individual)
  - results/ablation_spaces_hard.png (individual)
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load results
with open("results/ablation_spaces.json", "r") as f:
    data = json.load(f)

# Configuration
palette = {
    "evolved": "#1976D2",
    "obs_only": "#388E3C",
    "reward_only": "#F57C00",
    "sparse": "#D32F2F",
}

nice_label = {
    "evolved": "Full (Ours)",
    "obs_only": "Obs-Only",
    "reward_only": "Reward-Only",
    "sparse": "Sparse",
}


def plot_single_task(task_key, save_path):
    """Create individual plot for a single task."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    
    r = data[task_key]
    ts_per_update = r["num_envs"] * r["num_steps"]
    total_ts = r["timesteps"]

    for variant in ["evolved", "obs_only", "reward_only", "sparse"]:
        curve = r[variant]["success_curve"]
        std_curve = r[variant]["std_curve"]
        xs = [(i + 1) * ts_per_update / 1e6 for i in range(len(curve))]
        ys = [v * 100 for v in curve]
        std_ys = [s * 100 for s in std_curve]

        ax.plot(xs, ys, color=palette[variant], label=nice_label[variant], linewidth=2.2, alpha=0.9)
        ax.fill_between(
            xs,
            [max(0, y - s) for y, s in zip(ys, std_ys)],
            [min(100, y + s) for y, s in zip(ys, std_ys)],
            color=palette[variant], alpha=0.2,
        )

    ax.set_title(r["name"], fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Training Timesteps (millions)", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, total_ts / 1e6 + 0.1)
    ax.grid(True, alpha=0.25)
    
    # Legend at bottom
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=11, frameon=True, ncol=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


# Combined 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Evolved vs Ablation Spaces vs Sparse (10 seeds, mean ± std)", fontsize=14, fontweight="bold", y=1.02)

for ax, task_key in zip(axes, ["easy", "medium", "hard"]):
    r = data[task_key]
    ts_per_update = r["num_envs"] * r["num_steps"]
    total_ts = r["timesteps"]

    for variant in ["evolved", "obs_only", "reward_only", "sparse"]:
        curve = r[variant]["success_curve"]
        std_curve = r[variant]["std_curve"]
        xs = [(i + 1) * ts_per_update / 1e6 for i in range(len(curve))]
        ys = [v * 100 for v in curve]
        std_ys = [s * 100 for s in std_curve]

        ax.plot(xs, ys, color=palette[variant], label=nice_label[variant], linewidth=2.2, alpha=0.9)
        ax.fill_between(
            xs,
            [max(0, y - s) for y, s in zip(ys, std_ys)],
            [min(100, y + s) for y, s in zip(ys, std_ys)],
            color=palette[variant], alpha=0.2,
        )

    ax.set_title(r["name"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Timesteps (millions)", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, total_ts / 1e6 + 0.1)
    ax.grid(True, alpha=0.25)

# Single legend for combined figure (bottom, centered)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=True, ncol=4)

plt.tight_layout()
plt.savefig("results/ablation_spaces_combined.png", dpi=150, bbox_inches="tight")
print("Saved: results/ablation_spaces_combined.png")
plt.close()

# Individual plots
plot_single_task("easy", "results/ablation_spaces_easy.png")
plot_single_task("medium", "results/ablation_spaces_medium.png")
plot_single_task("hard", "results/ablation_spaces_hard.png")

print("\n✓ All plots generated!")
