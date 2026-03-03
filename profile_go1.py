"""Profile Go1 learning curve with eval every 2M steps."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import functools, sys, time
sys.path.insert(0, "mujoco_playground")

import jax
import jax.numpy as jp
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground._src import wrapper
from test_go1_pushrecovery import Go1PushRecovery

env = Go1PushRecovery()
eval_env = Go1PushRecovery()
print(f"Go1 action_size={env.action_size}, obs_size={env.observation_size}")

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_obs_key="state",
)

t0 = time.time()
num_timesteps = 50_000_000

def progress(num_steps, metrics):
    elapsed = time.time() - t0
    reward = metrics.get("eval/episode_reward", 0.0)
    success = metrics.get("eval/episode_success", 0.0)
    pos_err = metrics.get("eval/episode_pos_error", "?")
    print(f"  [{elapsed:7.1f}s] step={num_steps:>12,}  ({num_steps/1e6:.0f}M)  reward={reward:8.1f}  success={success:.3f}  pos_err={pos_err}")
    sys.stdout.flush()

train_fn = functools.partial(
    ppo.train,
    num_timesteps=num_timesteps,
    num_evals=25,  # eval every 2M steps
    reward_scaling=1.0,
    episode_length=500,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=256,
    max_grad_norm=1.0,
    network_factory=network_factory,
    seed=0,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)

print(f"\nTraining Go1PushRecovery for {num_timesteps/1e6:.0f}M steps (eval every 2M)...")
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress, eval_env=eval_env)
print(f"\nDone in {time.time()-t0:.1f}s")
