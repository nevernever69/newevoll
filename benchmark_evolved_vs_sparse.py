#!/usr/bin/env python3
"""Compare evolved MDP interface vs sparse baseline across training timesteps.

Trains both interfaces at multiple timestep checkpoints (50k, 100k, ..., 1M)
and plots success rate learning curves.

Sparse baseline = full grid observation (flattened) + task-completion-only reward.

Usage:
    python benchmark_evolved_vs_sparse.py
    python benchmark_evolved_vs_sparse.py --max-timesteps 500000 --step 50000 --seeds 3
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import xminigrid

from mdp_discovery.config import Config
from mdp_discovery.mdp_interface import MDPInterface
from mdp_discovery.train import run_training

# ---------------------------------------------------------------------------
# Sparse baseline: full grid obs, task-completion reward only
# ---------------------------------------------------------------------------
SPARSE_CODE = '''
import jax
import jax.numpy as jnp

def get_observation(state):
    """Full grid observation flattened into a 1D vector."""
    grid = state.grid.astype(jnp.float32)
    flat_grid = grid.flatten()

    # Agent position and direction
    agent_pos = state.agent.position.astype(jnp.float32)
    agent_dir = jax.nn.one_hot(state.agent.direction, 4, dtype=jnp.float32)

    return jnp.concatenate([flat_grid, agent_pos, agent_dir])

def compute_reward(state, action, next_state):
    """Sparse reward: +1.0 when agent reaches the goal tile, 0.0 otherwise."""
    pos = next_state.agent.position
    tile = next_state.grid[pos[0], pos[1], 0]
    on_goal = (tile == 6).astype(jnp.float32)  # GOAL tile ID = 6
    return on_goal
'''

# ---------------------------------------------------------------------------
# Evolved interface (best from the run)
# ---------------------------------------------------------------------------
EVOLVED_CODE = '''
import jax
import jax.numpy as jnp

def get_observation(state):
    """Simple but effective observation: agent position, direction, and goal direction."""
    H, W = state.grid.shape[:2]

    # Agent position normalized to [0, 1]
    agent_pos = state.agent.position.astype(jnp.float32)
    norm_pos = agent_pos / jnp.array([H - 1, W - 1], dtype=jnp.float32)

    # Agent direction as one-hot
    dir_onehot = jax.nn.one_hot(state.agent.direction, 4, dtype=jnp.float32)

    # Find goal position
    goal_mask = (state.grid[:, :, 0] == 6)
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    goal_count = jnp.maximum(jnp.sum(goal_mask), 1.0)
    goal_y = jnp.sum(yy * goal_mask) / goal_count
    goal_x = jnp.sum(xx * goal_mask) / goal_count

    # Goal position normalized
    norm_goal = jnp.array([goal_y, goal_x]) / jnp.array([H - 1, W - 1], dtype=jnp.float32)

    # Direction vector to goal (unnormalized)
    goal_direction = norm_goal - norm_pos

    # Distance to goal (normalized by max possible distance)
    max_dist = jnp.sqrt(2.0)
    goal_distance = jnp.sqrt(jnp.sum(goal_direction ** 2)) / max_dist

    # Simple local view: tiles in 4 cardinal directions from agent
    DIRECTIONS = jnp.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
    local_tiles = jnp.zeros(4, dtype=jnp.float32)

    for i in range(4):
        dy, dx = DIRECTIONS[i]
        neighbor_y = jnp.clip(agent_pos[0] + dy, 0, H - 1).astype(jnp.int32)
        neighbor_x = jnp.clip(agent_pos[1] + dx, 0, W - 1).astype(jnp.int32)

        in_bounds = ((agent_pos[0] + dy >= 0) & (agent_pos[0] + dy < H) &
                    (agent_pos[1] + dx >= 0) & (agent_pos[1] + dx < W))

        tile_id = jnp.where(in_bounds,
                           state.grid[neighbor_y, neighbor_x, 0].astype(jnp.float32),
                           2.0)
        local_tiles = local_tiles.at[i].set(tile_id / 12.0)

    observation = jnp.concatenate([
        norm_pos,
        dir_onehot,
        norm_goal,
        goal_direction,
        jnp.array([goal_distance]),
        local_tiles
    ])

    return observation

def compute_reward(state, action, next_state):
    """Simple distance-based reward for navigation."""
    goal_mask = (state.grid[:, :, 0] == 6)
    H, W = state.grid.shape[:2]
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    goal_count = jnp.maximum(jnp.sum(goal_mask), 1.0)
    goal_y = jnp.sum(yy * goal_mask) / goal_count
    goal_x = jnp.sum(xx * goal_mask) / goal_count

    prev_pos = state.agent.position.astype(jnp.float32)
    curr_pos = next_state.agent.position.astype(jnp.float32)

    prev_dist = jnp.abs(prev_pos[0] - goal_y) + jnp.abs(prev_pos[1] - goal_x)
    curr_dist = jnp.abs(curr_pos[0] - goal_y) + jnp.abs(curr_pos[1] - goal_x)

    reward = prev_dist - curr_dist

    return reward.astype(jnp.float32)
'''


def detect_obs_dim(code: str, config: Config) -> int:
    """Load interface and detect obs_dim from a dummy state."""
    interface = MDPInterface.from_code(code)
    env, env_params = xminigrid.make(config.environment.env_id)
    key = jax.random.key(0)
    timestep = env.reset(env_params, key)
    return interface.validate(timestep.state)


def run_at_timestep(code: str, config: Config, obs_dim: int, timesteps: int) -> dict:
    """Train one interface for a given number of timesteps, return metrics."""
    interface = MDPInterface.from_code(code)
    interface.obs_dim = obs_dim
    metrics = run_training(config, interface, obs_dim, total_timesteps=timesteps)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evolved vs Sparse MDP interface comparison")
    parser.add_argument("--max-timesteps", type=int, default=1_000_000, help="Max training timesteps")
    parser.add_argument("--step", type=int, default=50_000, help="Timestep interval between evaluations")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per configuration")
    parser.add_argument("--output", type=str, default="benchmark_evolved_vs_sparse.png", help="Output plot")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    # Compute valid timestep checkpoints (must produce at least 1 update)
    tc = config.training
    num_devices = jax.local_device_count()
    min_timesteps = tc.num_steps * tc.num_envs  # minimum for 1 update
    timestep_points = list(range(args.step, args.max_timesteps + 1, args.step))
    timestep_points = [t for t in timestep_points if t >= min_timesteps]

    if not timestep_points:
        print(f"ERROR: step size {args.step} too small. Minimum timesteps = {min_timesteps}")
        return

    print(f"{'='*60}")
    print(f"Evolved vs Sparse MDP Interface Comparison")
    print(f"{'='*60}")
    print(f"  Timestep checkpoints: {timestep_points}")
    print(f"  Seeds per config:     {args.seeds}")
    print(f"  Devices:              {num_devices}")
    print(f"  Num envs:             {tc.num_envs}")
    print(f"  Min timesteps:        {min_timesteps}")
    print(f"{'='*60}\n")

    # Detect obs dims
    sparse_obs_dim = detect_obs_dim(SPARSE_CODE, config)
    evolved_obs_dim = detect_obs_dim(EVOLVED_CODE, config)
    print(f"Sparse obs_dim:  {sparse_obs_dim}")
    print(f"Evolved obs_dim: {evolved_obs_dim}\n")

    interfaces = {
        "Evolved (shaped reward + compact obs)": (EVOLVED_CODE, evolved_obs_dim),
        "Sparse (full grid obs + completion reward)": (SPARSE_CODE, sparse_obs_dim),
    }

    all_results = {}

    for name, (code, obs_dim) in interfaces.items():
        print(f"\n{'='*60}")
        print(f"Training: {name} (obs_dim={obs_dim})")
        print(f"{'='*60}")

        results_per_seed = []
        for seed in range(args.seeds):
            config.training.seed = 42 + seed
            seed_results = []

            for ts in timestep_points:
                print(f"  [seed={seed}] timesteps={ts:>10,} ... ", end="", flush=True)
                t0 = time.time()
                metrics = run_at_timestep(code, config, obs_dim, ts)
                elapsed = time.time() - t0
                print(f"success={metrics['success_rate']*100:.0f}%  ({elapsed:.1f}s)")
                seed_results.append({
                    "timesteps": ts,
                    "success_rate": metrics["success_rate"],
                    "final_return": metrics["final_return"],
                    "final_length": metrics["final_length"],
                    "training_time": metrics["training_time"],
                    "success_curve": metrics["success_curve"],
                })

            results_per_seed.append(seed_results)

        all_results[name] = results_per_seed

    # Reset seed
    config.training.seed = 42

    # Save raw data
    json_path = args.output.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw data saved to {json_path}")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Evolved MDP Interface vs Sparse Baseline", fontsize=14, fontweight="bold")

    colors = {"Evolved (shaped reward + compact obs)": "#2196F3",
              "Sparse (full grid obs + completion reward)": "#F44336"}

    # --- Panel 1: Success rate vs timesteps ---
    ax = axes[0]
    for name, results_per_seed in all_results.items():
        # results_per_seed: list of seeds, each a list of {timesteps, success_rate, ...}
        n_seeds = len(results_per_seed)
        n_points = len(timestep_points)

        matrix = np.zeros((n_seeds, n_points))
        for s, seed_results in enumerate(results_per_seed):
            for i, r in enumerate(seed_results):
                matrix[s, i] = r["success_rate"] * 100

        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        xs = [t / 1000 for t in timestep_points]  # in thousands

        color = colors[name]
        ax.plot(xs, mean, "o-", color=color, label=name, linewidth=2, markersize=5)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Training Timesteps (thousands)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs Training Timesteps")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # --- Panel 2: Bar chart at final timestep ---
    ax = axes[1]
    final_ts = timestep_points[-1]
    bar_data = []
    bar_labels = []
    bar_errs = []
    bar_colors = []

    for name, results_per_seed in all_results.items():
        rates = [seed[-1]["success_rate"] * 100 for seed in results_per_seed]
        bar_data.append(np.mean(rates))
        bar_errs.append(np.std(rates))
        bar_labels.append(name.split("(")[0].strip())
        bar_colors.append(colors[name])

    bars = ax.bar(bar_labels, bar_data, yerr=bar_errs, color=bar_colors,
                  width=0.5, capsize=8, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, bar_data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"Final Success Rate @ {final_ts/1000:.0f}k Steps")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {args.output}")
    plt.close()

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Interface':<45} {'Success':>10} {'Std':>8}")
    print(f"{'='*70}")
    for name, results_per_seed in all_results.items():
        rates = [seed[-1]["success_rate"] * 100 for seed in results_per_seed]
        print(f"{name:<45} {np.mean(rates):>9.1f}% {np.std(rates):>7.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
