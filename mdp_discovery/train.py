"""PPO training for MDP Interface Discovery.

Adapted from xland-minigrid's train_single_task.py.
No wandb dependency — returns metrics as a dictionary.
"""

import time
from functools import partial
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState

from mdp_discovery.adapters.base import EnvAdapter
from mdp_discovery.config import Config
from mdp_discovery.mdp_interface import MDPInterface
from mdp_discovery.nn import ActorCriticMLP
from mdp_discovery.train_utils import (
    Transition,
    calculate_gae,
    ppo_update_networks,
    rollout,
)

jax.config.update("jax_threefry_partitionable", True)


def _compute_device_splits(config: Config, total_timesteps: int):
    """Compute per-device splits for the training loop."""
    num_devices = jax.local_device_count()
    tc = config.training

    num_envs_per_device = tc.num_envs // num_devices
    total_timesteps_per_device = total_timesteps // num_devices
    eval_episodes_per_device = max(tc.eval_episodes // num_devices, 1)
    num_updates = total_timesteps_per_device // tc.num_steps // num_envs_per_device

    assert tc.num_envs % num_devices == 0, (
        f"num_envs ({tc.num_envs}) must be divisible by num_devices ({num_devices})"
    )
    assert num_updates > 0, (
        f"total_timesteps too low: need at least {tc.num_steps * num_envs_per_device * num_devices}"
    )
    return num_envs_per_device, total_timesteps_per_device, eval_episodes_per_device, num_updates


def make_states(
    config: Config,
    adapter: EnvAdapter,
    mdp_interface: MDPInterface,
    obs_dim: int,
    total_timesteps: int,
):
    """Set up environment, network, and training state."""
    tc = config.training
    num_envs_per_device, _, _, num_updates = _compute_device_splits(config, total_timesteps)

    # --- Learning rate schedule ---
    def linear_schedule(count):
        frac = 1.0 - (count // (tc.num_minibatches * tc.update_epochs)) / num_updates
        return tc.lr * frac

    # --- Environment with MDP interface wrapper ---
    env, env_params = adapter.make_env(
        mdp_interface.get_observation,
        mdp_interface.compute_reward,
    )

    # --- Network ---
    rng = jax.random.key(tc.seed)
    rng, _rng = jax.random.split(rng)

    network = ActorCriticMLP(
        num_actions=adapter.num_actions(env_params),
        hidden_dim=tc.hidden_dim,
        action_emb_dim=tc.action_emb_dim,
        rnn_hidden_dim=tc.rnn_hidden_dim,
        rnn_num_layers=tc.rnn_num_layers,
        head_hidden_dim=tc.head_hidden_dim,
        dtype=jnp.bfloat16 if tc.enable_bf16 else None,
    )

    init_obs = {
        "obs": jnp.zeros((num_envs_per_device, 1, obs_dim)),
        "prev_action": jnp.zeros((num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((num_envs_per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=num_envs_per_device)

    network_params = network.init(_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(tc.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=linear_schedule, eps=1e-8
        ),
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx
    )

    return rng, env, env_params, init_hstate, train_state


def make_train(env, env_params, config: Config, total_timesteps: int, compute_success_fn):
    """Create the JIT-compiled, pmap'd training function.

    Args:
        env: Wrapped environment.
        env_params: Environment parameters.
        config: System configuration.
        total_timesteps: Total training timesteps.
        compute_success_fn: Pure JAX function (rollout_stats, env_params) -> float array.
    """
    tc = config.training
    num_envs_per_device, _, eval_episodes_per_device, num_updates = _compute_device_splits(
        config, total_timesteps
    )

    @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        train_state: TrainState,
        init_hstate: jax.Array,
    ):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs_per_device)
        timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
        prev_action = jnp.zeros(num_envs_per_device, dtype=jnp.int32)
        prev_reward = jnp.zeros(num_envs_per_device)

        # TRAIN LOOP
        def _update_step(runner_state, _):
            def _env_step(runner_state, _):
                rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state

                rng, _rng = jax.random.split(rng)
                dist, value, hstate = train_state.apply_fn(
                    train_state.params,
                    {
                        "obs": prev_timestep.observation[:, None],
                        "prev_action": prev_action[:, None],
                        "prev_reward": prev_reward[:, None],
                    },
                    prev_hstate,
                )
                action, log_prob = dist.sample_and_log_prob(seed=_rng)
                action, value, log_prob = (
                    action.squeeze(1),
                    value.squeeze(1),
                    log_prob.squeeze(1),
                )

                timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(
                    env_params, prev_timestep, action
                )
                transition = Transition(
                    done=timestep.last(),
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                )
                runner_state = (
                    rng, train_state, timestep, action, timestep.reward, hstate
                )
                return runner_state, transition

            initial_hstate = runner_state[-1]
            runner_state, transitions = jax.lax.scan(
                _env_step, runner_state, None, tc.num_steps
            )

            # CALCULATE ADVANTAGE
            rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
            _, last_val, _ = train_state.apply_fn(
                train_state.params,
                {
                    "obs": timestep.observation[:, None],
                    "prev_action": prev_action[:, None],
                    "prev_reward": prev_reward[:, None],
                },
                hstate,
            )
            advantages, targets = calculate_gae(
                transitions, last_val.squeeze(1), tc.gamma, tc.gae_lambda
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, transitions, advantages, targets = batch_info
                    return ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        init_hstate=init_hstate.squeeze(1),
                        advantages=advantages,
                        targets=targets,
                        clip_eps=tc.clip_eps,
                        vf_coef=tc.vf_coef,
                        ent_coef=tc.ent_coef,
                    )

                rng, train_state, init_hstate, transitions, advantages, targets = (
                    update_state
                )

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, num_envs_per_device)
                batch = (init_hstate, transitions, advantages, targets)
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)
                shuffled_batch = jtu.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(
                        x, (tc.num_minibatches, -1) + x.shape[1:]
                    ),
                    shuffled_batch,
                )
                train_state, update_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    rng, train_state, init_hstate, transitions, advantages, targets
                )
                return update_state, update_info

            init_hstate = initial_hstate[None, :]
            update_state = (
                rng, train_state, init_hstate, transitions, advantages, targets
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, tc.update_epochs
            )
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            rng, train_state = update_state[:2]

            # EVALUATE
            rng, _rng = jax.random.split(rng)
            eval_rng = jax.random.split(_rng, num=eval_episodes_per_device)
            eval_stats = jax.vmap(rollout, in_axes=(0, None, None, None, None, None))(
                eval_rng,
                env,
                env_params,
                train_state,
                jnp.zeros(
                    (1, tc.rnn_num_layers, tc.rnn_hidden_dim),
                    dtype=jnp.bfloat16 if tc.enable_bf16 else None,
                ),
                1,
            )
            # Compute success rate per device before pmean
            per_ep_success = compute_success_fn(eval_stats, env_params)
            success_rate = per_ep_success.mean(0)
            success_rate = jax.lax.pmean(success_rate, axis_name="devices")

            eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")
            loss_info.update(
                {
                    "eval/returns": eval_stats.reward.mean(0),
                    "eval/lengths": eval_stats.length.mean(0),
                    "eval/success_rate": success_rate,
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                }
            )
            runner_state = (
                rng, train_state, timestep, prev_action, prev_reward, hstate
            )
            return runner_state, loss_info

        runner_state = (rng, train_state, timestep, prev_action, prev_reward, init_hstate)
        runner_state, loss_info = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "loss_info": loss_info}

    return train


def _run_success_eval(
    config: Config,
    adapter: EnvAdapter,
    mdp_interface: MDPInterface,
    obs_dim: int,
    train_state: TrainState,
    num_episodes: int = 50,
) -> float:
    """Run final evaluation: percentage of episodes where the agent reaches the goal.

    Uses the full MDPInterfaceWrapper (so the agent sees the same observations
    and prev_reward it was trained with), but measures success via the adapter's
    compute_success method.

    Args:
        config: Full system configuration.
        adapter: Environment adapter.
        mdp_interface: Loaded MDP interface.
        obs_dim: Observation dimension.
        train_state: Trained agent (unreplicated, single-device).
        num_episodes: Number of evaluation episodes.

    Returns:
        Success rate in [0.0, 1.0].
    """
    tc = config.training

    # Create env with full wrapper (agent needs evolved obs + reward for prev_reward)
    eval_max_steps = tc.eval_max_steps
    env, env_params = adapter.make_eval_env(
        mdp_interface.get_observation,
        mdp_interface.compute_reward,
        max_steps=eval_max_steps,
    )
    max_steps = eval_max_steps

    rng = jax.random.key(tc.seed + 99999)
    eval_rngs = jax.random.split(rng, num_episodes)

    init_hstate = jnp.zeros(
        (1, tc.rnn_num_layers, tc.rnn_hidden_dim),
        dtype=jnp.bfloat16 if tc.enable_bf16 else None,
    )

    @jax.jit
    def _eval_batch(rngs):
        return jax.vmap(rollout, in_axes=(0, None, None, None, None, None))(
            rngs, env, env_params, train_state, init_hstate, 1
        )

    stats = jax.block_until_ready(_eval_batch(eval_rngs))
    successes = int(adapter.compute_success(stats, env_params).sum())
    return successes / num_episodes


def run_training(
    config: Config,
    adapter: EnvAdapter,
    mdp_interface: MDPInterface,
    obs_dim: int,
    total_timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run PPO training and return metrics.

    Args:
        config: Full system configuration.
        adapter: Environment adapter for env creation and success computation.
        mdp_interface: Loaded MDP interface with get_observation and/or compute_reward.
        obs_dim: Observation dimension (detected by crash filter).
        total_timesteps: Override total timesteps (for cascade evaluation stages).

    Returns:
        Dictionary with:
            final_return: Mean eval return at the end of training.
            final_length: Mean eval episode length at the end of training.
            learning_curve: List of eval returns at each update step.
            training_time: Wall-clock seconds.
    """
    if total_timesteps is None:
        total_timesteps = config.training.total_timesteps

    rng, env, env_params, init_hstate, train_state = make_states(
        config, adapter, mdp_interface, obs_dim, total_timesteps
    )

    # Replicate across devices
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())

    # Extract a pure JAX success function from the adapter
    compute_success_fn = adapter.compute_success

    # Build pmap'd training function
    train_fn = make_train(env, env_params, config, total_timesteps, compute_success_fn)

    # Train (JIT compilation happens on first call)
    t = time.time()
    train_info = jax.block_until_ready(train_fn(rng, train_state, init_hstate))
    training_time = time.time() - t

    # Extract metrics
    loss_info = unreplicate(train_info["loss_info"])
    returns = loss_info["eval/returns"]
    lengths = loss_info["eval/lengths"]
    success_rates = loss_info["eval/success_rate"]

    # Final dedicated success rate evaluation
    final_train_state = unreplicate(train_info["runner_state"][1])
    final_success_rate = _run_success_eval(
        config, adapter, mdp_interface, obs_dim, final_train_state,
        num_episodes=config.training.eval_episodes_for_fitness,
    )

    return {
        "final_return": float(returns[-1]),
        "final_length": float(lengths[-1]),
        "success_rate": final_success_rate,
        "learning_curve": [float(r) for r in returns],
        "success_curve": [float(s) for s in success_rates],
        "training_time": training_time,
    }
