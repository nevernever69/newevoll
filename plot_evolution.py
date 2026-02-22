"""
Evolution progress figure — publication quality (NeurIPS style).

Uses evolution_trace.jsonl + evolution.log for per-iteration data.
White background, serif fonts, minimal chrome.
"""

import re
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── NeurIPS-style defaults ──────────────────────────────────────────────
sns.set_theme(style="white", context="paper", font_scale=1.15)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.labelpad": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.5,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
})

# Colors — standard matplotlib / colorblind-friendly
C_EASY   = "#2ca02c"
C_MEDIUM = "#1f77b4"
C_HARD   = "#d62728"

TASKS = [
    ("easy",   "experiments/main/easy",   C_EASY,
     "(a) Easy — Pick Up Blue Pyramid"),
    ("medium", "experiments/main/medium", C_MEDIUM,
     "(b) Medium — Place Pyramid Adjacent to Square"),
    ("hard",   "experiments/main/hard",   C_HARD,
     "(c) Hard — Rule Chain + Place Near Goal (4 rooms)"),
]


def parse_evolution_log(log_path):
    """Parse evolution.log → list of dicts with iter, sr, seeds, rejected."""
    candidates = []
    seed_buf = []  # running buffer of seed values

    with open(log_path) as f:
        for line in f:
            text = line.strip()

            # Seed line
            sm = re.search(r"Seed (\d+)/(\d+): success=(\d+)%", text)
            if sm:
                seed_buf.append(int(sm.group(3)) / 100.0)
                continue

            # Multi-seed average → flush seeds for this candidate
            am = re.search(r"Multi-seed avg.*?success=(\d+)%", text)
            if am:
                avg = int(am.group(1)) / 100.0
                # Find best matching 3 seeds from buffer
                # (handles interleaved parallel candidates)
                n_seeds = int(re.search(r"(\d+) seeds", text).group(1))
                # Take best-matching subset of size n_seeds
                matched = _match_seeds(seed_buf, avg, n_seeds)
                # Remove matched seeds from buffer
                for s in matched:
                    seed_buf.remove(s)
                # Store temporarily until we see the Iter line
                candidates.append({"seeds": matched, "avg": avg})
                continue

            # Iter result line
            im = re.search(r"\[Iter (\d+)\].*?success=(\d+)%", text)
            if im:
                it = int(im.group(1))
                sr = int(im.group(2)) / 100.0
                rejected = "REJECTED" in text
                if not rejected and candidates and "iter" not in candidates[-1]:
                    candidates[-1]["iter"] = it
                    candidates[-1]["sr"] = sr
                    candidates[-1]["rejected"] = False
                else:
                    candidates.append({
                        "iter": it, "sr": sr, "seeds": [],
                        "rejected": rejected, "avg": sr,
                    })

    # Filter to only those with iter assigned
    return [c for c in candidates if "iter" in c]


def _match_seeds(buf, avg, n):
    """Find n seeds from buf that best match the given average."""
    if len(buf) <= n:
        return list(buf)
    # Try all combinations of n from buf
    from itertools import combinations
    best = None
    best_err = float("inf")
    for combo in combinations(range(len(buf)), n):
        vals = [buf[i] for i in combo]
        err = abs(np.mean(vals) - avg)
        if err < best_err:
            best_err = err
            best = vals
    return best if best else buf[-n:]


# ── Build figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(5.5, 7.0))

for idx, (task, base_dir, color, title) in enumerate(TASKS):
    ax = axes[idx]
    log_path = f"{base_dir}/evolution.log"
    candidates = parse_evolution_log(log_path)

    iters = np.array([c["iter"] for c in candidates])
    srs = np.array([c["sr"] for c in candidates])
    rejected = np.array([c["rejected"] for c in candidates])

    # Running best (cumulative max)
    running_best = np.maximum.accumulate(srs)

    # ── Plot individual seeds as small jittered dots ──
    for c in candidates:
        if c["seeds"] and not c["rejected"]:
            jitter = np.linspace(-0.2, 0.2, len(c["seeds"]))
            seed_vals = np.array(c["seeds"]) * 100
            ax.scatter(
                c["iter"] + jitter, seed_vals,
                color=color, alpha=0.18, s=10, zorder=2,
                edgecolors="none", marker="o",
            )

    # ── Plot averaged candidate dots ──
    # Rejected: gray × marks
    rej_mask = rejected
    if rej_mask.any():
        ax.scatter(
            iters[rej_mask], srs[rej_mask] * 100,
            color="#999999", alpha=0.35, s=15, zorder=3,
            marker="x", linewidths=0.7,
        )

    # Accepted: colored dots
    acc_mask = ~rejected
    if acc_mask.any():
        ax.scatter(
            iters[acc_mask], srs[acc_mask] * 100,
            color=color, alpha=0.55, s=22, zorder=4,
            edgecolors="white", linewidths=0.3, marker="o",
        )

    # ── Running best step line ──
    ax.step(iters, running_best * 100, where="post", color=color,
            linewidth=1.8, zorder=5, label="Best so far")

    # Light fill
    ax.fill_between(iters, 0, running_best * 100, step="post",
                     color=color, alpha=0.05, zorder=1)

    # ── Annotate final best ──
    best_val = running_best[-1] * 100
    best_iter = iters[np.argmax(srs)]
    # Place annotation to avoid overlap
    x_off = -28 if best_iter > max(iters) * 0.7 else 10
    y_off = -12 if best_val > 85 else 8
    ax.annotate(
        f"{best_val:.0f}%",
        xy=(best_iter, best_val),
        xytext=(x_off, y_off), textcoords="offset points",
        fontsize=9, fontweight="bold", color=color,
        ha="left",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor=color, linewidth=0.6),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.6,
                        connectionstyle="arc3,rad=0.1"),
        zorder=7,
    )

    # ── Axes ──
    ax.set_ylabel("Success Rate (%)", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left", pad=5)
    ax.set_ylim(-3, 108)
    ax.set_xlim(0, max(iters) + 1.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(12.5))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax.tick_params(labelsize=8)
    sns.despine(ax=ax)
    ax.grid(axis="y", linewidth=0.3, alpha=0.35)

    # Custom legend
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection
    handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label="Best so far"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markersize=4.5, alpha=0.55, label="Candidate (avg)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markersize=3, alpha=0.2, label="Individual seeds"),
        Line2D([0], [0], marker="x", color="#999999", markersize=4,
               linewidth=0, alpha=0.5, label="Rejected"),
    ]
    ax.legend(handles=handles, loc="lower right" if task != "easy" else "center right",
              fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.3)

axes[-1].set_xlabel("Candidate Evaluation (chronological)", fontsize=9)

plt.tight_layout(h_pad=1.0)
plt.savefig("results/evolution_progress.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none", pad_inches=0.08)
plt.close()
print("Saved to results/evolution_progress.png")
