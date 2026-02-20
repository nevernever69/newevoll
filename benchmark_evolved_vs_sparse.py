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
from mdp_discovery.adapters.xminigrid_adapter import XMinigridAdapter

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

    # Pocket contents
    pocket = state.agent.pocket.astype(jnp.float32)

    return jnp.concatenate([flat_grid, agent_pos, agent_dir, pocket])

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
    """Extract observation for DoorKey task."""
    H, W = state.grid.shape[:2]

    # Agent state (normalized)
    agent_y = state.agent.position[0].astype(jnp.float32) / (H - 1)
    agent_x = state.agent.position[1].astype(jnp.float32) / (W - 1)
    agent_dir = jax.nn.one_hot(state.agent.direction, 4).astype(jnp.float32)

    # Pocket state
    has_key = (state.agent.pocket[0] == 7).astype(jnp.float32)
    pocket_empty = (state.agent.pocket[0] == 0).astype(jnp.float32)

    # Grid encoding - flatten and normalize
    grid_tiles = state.grid[:, :, 0].astype(jnp.float32) / 12.0
    grid_colors = state.grid[:, :, 1].astype(jnp.float32) / 11.0

    # Key position
    key_mask = (state.grid[:, :, 0] == 7)
    ys = jnp.arange(H).astype(jnp.float32)
    xs = jnp.arange(W).astype(jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    key_exists = jnp.sum(key_mask) > 0
    key_y = jnp.sum(yy * key_mask) / jnp.maximum(jnp.sum(key_mask), 1)
    key_x = jnp.sum(xx * key_mask) / jnp.maximum(jnp.sum(key_mask), 1)
    key_y_norm = key_y / (H - 1)
    key_x_norm = key_x / (W - 1)

    # Goal position
    goal_mask = (state.grid[:, :, 0] == 6)
    goal_exists = jnp.sum(goal_mask) > 0
    goal_y = jnp.sum(yy * goal_mask) / jnp.maximum(jnp.sum(goal_mask), 1)
    goal_x = jnp.sum(xx * goal_mask) / jnp.maximum(jnp.sum(goal_mask), 1)
    goal_y_norm = goal_y / (H - 1)
    goal_x_norm = goal_x / (W - 1)

    # Locked door position
    door_mask = (state.grid[:, :, 0] == 8)
    door_exists = jnp.sum(door_mask) > 0
    door_y = jnp.sum(yy * door_mask) / jnp.maximum(jnp.sum(door_mask), 1)
    door_x = jnp.sum(xx * door_mask) / jnp.maximum(jnp.sum(door_mask), 1)
    door_y_norm = door_y / (H - 1)
    door_x_norm = door_x / (W - 1)

    # Distances
    key_dist = (jnp.abs(agent_y * (H - 1) - key_y) + jnp.abs(agent_x * (W - 1) - key_x)) / (H + W - 2)
    door_dist = (jnp.abs(agent_y * (H - 1) - door_y) + jnp.abs(agent_x * (W - 1) - door_x)) / (H + W - 2)
    goal_dist = (jnp.abs(agent_y * (H - 1) - goal_y) + jnp.abs(agent_x * (W - 1) - goal_x)) / (H + W - 2)

    step_norm = state.step_num.astype(jnp.float32) / 100.0

    # What is in front of agent
    DIRECTIONS = jnp.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
    dy, dx = DIRECTIONS[state.agent.direction]
    front_y = jnp.clip(state.agent.position[0] + dy, 0, H - 1)
    front_x = jnp.clip(state.agent.position[1] + dx, 0, W - 1)
    front_tile = state.grid[front_y, front_x, 0].astype(jnp.float32) / 12.0
    front_color = state.grid[front_y, front_x, 1].astype(jnp.float32) / 11.0

    basic_features = jnp.array([
        agent_y, agent_x, has_key, pocket_empty,
        key_y_norm, key_x_norm, key_exists.astype(jnp.float32),
        goal_y_norm, goal_x_norm, goal_exists.astype(jnp.float32),
        door_y_norm, door_x_norm, door_exists.astype(jnp.float32),
        key_dist, door_dist, goal_dist,
        step_norm, front_tile, front_color
    ])

    grid_flat = jnp.concatenate([grid_tiles.flatten(), grid_colors.flatten()])

    observation = jnp.concatenate([
        basic_features,
        agent_dir,
        grid_flat
    ]).astype(jnp.float32)

    return observation

def compute_reward(state, action, next_state):
    """Compute reward for DoorKey task."""
    H, W = state.grid.shape[:2]

    agent_pos = next_state.agent.position
    goal_tile = next_state.grid[agent_pos[0], agent_pos[1], 0]
    goal_reached = (goal_tile == 6).astype(jnp.float32)

    goal_reward = goal_reached * 10.0

    had_key = (state.agent.pocket[0] == 7).astype(jnp.float32)
    has_key = (next_state.agent.pocket[0] == 7).astype(jnp.float32)
    key_pickup_reward = (has_key - had_key) * 3.0

    door_was_locked = jnp.sum(state.grid[:, :, 0] == 8) > 0
    door_is_locked = jnp.sum(next_state.grid[:, :, 0] == 8) > 0
    door_unlocked = (door_was_locked & ~door_is_locked).astype(jnp.float32)
    unlock_reward = door_unlocked * 5.0

    step_penalty = -0.01

    key_mask = (next_state.grid[:, :, 0] == 7)
    goal_mask = (next_state.grid[:, :, 0] == 6)
    door_mask = (next_state.grid[:, :, 0] == 8)

    ys = jnp.arange(H).astype(jnp.float32)
    xs = jnp.arange(W).astype(jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

    key_y = jnp.sum(yy * key_mask) / jnp.maximum(jnp.sum(key_mask), 1)
    key_x = jnp.sum(xx * key_mask) / jnp.maximum(jnp.sum(key_mask), 1)
    key_dist = jnp.abs(agent_pos[0] - key_y) + jnp.abs(agent_pos[1] - key_x)
    key_shaping = jax.lax.select(
        (has_key == 0) & (jnp.sum(key_mask) > 0),
        -0.1 * key_dist / (H + W),
        0.0
    )

    door_y = jnp.sum(yy * door_mask) / jnp.maximum(jnp.sum(door_mask), 1)
    door_x = jnp.sum(xx * door_mask) / jnp.maximum(jnp.sum(door_mask), 1)
    door_dist = jnp.abs(agent_pos[0] - door_y) + jnp.abs(agent_pos[1] - door_x)
    door_shaping = jax.lax.select(
        (has_key == 1) & (jnp.sum(door_mask) > 0),
        -0.1 * door_dist / (H + W),
        0.0
    )

    goal_y = jnp.sum(yy * goal_mask) / jnp.maximum(jnp.sum(goal_mask), 1)
    goal_x = jnp.sum(xx * goal_mask) / jnp.maximum(jnp.sum(goal_mask), 1)
    goal_dist = jnp.abs(agent_pos[0] - goal_y) + jnp.abs(agent_pos[1] - goal_x)
    goal_shaping = jax.lax.select(
        (jnp.sum(door_mask) == 0) & (goal_reached == 0),
        -0.1 * goal_dist / (H + W),
        0.0
    )

    total_reward = (goal_reward + key_pickup_reward + unlock_reward +
                   step_penalty + key_shaping + door_shaping + goal_shaping)

    return total_reward.astype(jnp.float32)
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
    adapter = XMinigridAdapter(config)
    metrics = run_training(config, adapter, interface, obs_dim, total_timesteps=timesteps)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evolved vs Sparse MDP interface comparison")
    parser.add_argument("--max-timesteps", type=int, default=1_000_000, help="Max training timesteps")
    parser.add_argument("--step", type=int, default=50_000, help="Timestep interval between evaluations")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per configuration")
    parser.add_argument("--output", type=str, default="benchmark_evolved_vs_sparse.png", help="Output plot")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--env", type=str, default=None, help="Override environment ID")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.env:
        config.environment.env_id = args.env

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
        "Evolved (shaped reward + grid obs)": (EVOLVED_CODE, evolved_obs_dim),
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

    colors = {"Evolved (shaped reward + grid obs)": "#2196F3",
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
