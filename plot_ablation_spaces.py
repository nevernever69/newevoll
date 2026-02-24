#!/usr/bin/env python3
"""Generate publication-quality plots from ablation spaces benchmark results.

Reads: results/ablation_spaces.json
Outputs: 
  - results/ablation_spaces_combined.pdf (3-panel figure)
  - results/ablation_spaces_easy.pdf (individual)
  - results/ablation_spaces_medium.pdf (individual)
  - results/ablation_spaces_hard.pdf (individual)
  - results/ablation_spaces_combined.png (web version)
  - results/ablation_spaces_{easy,medium,hard}.png (web versions)

Design principles for NeurIPS-quality figures:
- No annotations on curves (cleaner)
- Legend outside plot area (no occlusion)
- Shaded confidence intervals (±1 std)
- Consistent color scheme
- High DPI (300) for publication
- PDF format for vector graphics (infinite resolution)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# ─── Load results ─────────────────────────────────────────────────────────────
with open("results/ablation_spaces.json", "r") as f:
    data = json.load(f)

# ─── Style configuration ──────────────────────────────────────────────────────
# NeurIPS-friendly color palette (colorblind-safe)
PALETTE = {
    "evolved": "#1976D2",      # blue
    "obs_only": "#388E3C",     # green
    "reward_only": "#F57C00",  # orange
    "sparse": "#D32F2F",       # red
}

LABELS = {
    "evolved": "Full (Ours)",
    "obs_only": "Obs-Only",
    "reward_only": "Reward-Only",
    "sparse": "Sparse Baseline",
}

LINE_WIDTH = 2.5
ALPHA_FILL = 0.25
ALPHA_LINE = 0.95

# Font sizes for publication
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 11
FONT_SIZE_LEGEND = 11

# ─── Helper functions ─────────────────────────────────────────────────────────

def plot_task(ax, data, task_key, show_legend=False):
    """Plot a single task's results on given axes."""
    r = data[task_key]
    ts_per_update = r["num_envs"] * r["num_steps"]
    total_ts = r["timesteps"]
    
    xs = [(i + 1) * ts_per_update / 1e6 for i in range(len(r["evolved"]["success_curve"]))]
    
    for variant in ["evolved", "obs_only", "reward_only", "sparse"]:
        curve = r[variant]["success_curve"]
        std_curve = r[variant]["std_curve"]
        ys = [v * 100 for v in curve]
        std_ys = [s * 100 for s in std_curve]
        
        # Main line
        ax.plot(
            xs, ys,
            color=PALETTE[variant],
            label=LABELS[variant] if show_legend else "",
            linewidth=LINE_WIDTH,
            alpha=ALPHA_LINE,
        )
        # Shaded confidence interval
        ax.fill_between(
            xs,
            [max(0, y - s) for y, s in zip(ys, std_ys)],
            [min(100, y + s) for y, s in zip(ys, std_ys)],
            color=PALETTE[variant],
            alpha=ALPHA_FILL,
            linewidth=0,
        )
    
    ax.set_xlim(0, total_ts / 1e6)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Training Timesteps (millions)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Success Rate (%)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
    ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.5)
    
    # Set title
    task_name = r["name"].replace("\n", " ")
    ax.set_title(task_name, fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=10)


def create_combined_figure(data):
    """Create 3-panel combined figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, task_key in zip(axes, ["easy", "medium", "hard"]):
        plot_task(ax, data, task_key, show_legend=False)
    
    # Legend outside the figure (right side)
    handles = [
        plt.Line2D([0], [0], color=PALETTE[variant], linewidth=LINE_WIDTH, alpha=ALPHA_LINE)
        for variant in ["evolved", "obs_only", "reward_only", "sparse"]
    ]
    labels = [LABELS[variant] for variant in ["evolved", "obs_only", "reward_only", "sparse"]]
    
    fig.legend(
        handles, labels,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        fontsize=FONT_SIZE_LEGEND,
        frameon=False,
    )
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave space for legend on right
    
    return fig


def create_individual_figure(data, task_key):
    """Create individual figure for a single task."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    
    plot_task(ax, data, task_key, show_legend=True)
    
    # Legend at top-right, outside plot area
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        linewidth=0.5,
    )
    
    plt.tight_layout()
    
    return fig


# ─── Generate plots ───────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)

# Combined 3-panel figure
print("Generating combined 3-panel figure...")
fig_combined = create_combined_figure(data)
fig_combined.savefig("results/ablation_spaces_combined.pdf", bbox_inches="tight", dpi=300)
fig_combined.savefig("results/ablation_spaces_combined.png", bbox_inches="tight", dpi=150)
print("  → results/ablation_spaces_combined.pdf")
print("  → results/ablation_spaces_combined.png")
plt.close(fig_combined)

# Individual task figures
for task_key in ["easy", "medium", "hard"]:
    print(f"Generating individual figure for {task_key}...")
    fig_ind = create_individual_figure(data, task_key)
    fig_ind.savefig(f"results/ablation_spaces_{task_key}.pdf", bbox_inches="tight", dpi=300)
    fig_ind.savefig(f"results/ablation_spaces_{task_key}.png", bbox_inches="tight", dpi=150)
    print(f"  → results/ablation_spaces_{task_key}.pdf")
    print(f"  → results/ablation_spaces_{task_key}.png")
    plt.close(fig_ind)

print("\n✓ All plots generated successfully!")
print("\nFiles created:")
print("  Combined:  results/ablation_spaces_combined.pdf (.png)")
print("  Easy:      results/ablation_spaces_easy.pdf (.png)")
print("  Medium:    results/ablation_spaces_medium.pdf (.png)")
print("  Hard:      results/ablation_spaces_hard.pdf (.png)")
