"""Neural network for MDP Interface Discovery.

MLP-based actor-critic with GRU memory, designed for flat 1D observations
produced by the LLM-discovered get_observation function.

GRU / RNNModel / BatchedRNNModel are copied from xland-minigrid's
training/nn.py (not part of the pip package).
"""

import math
from typing import Optional, TypedDict

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from flax.typing import Dtype


# ---------------------------------------------------------------------------
# GRU / RNN (copied from xland-minigrid training/nn.py lines 18-72)
# ---------------------------------------------------------------------------


class GRU(nn.Module):
    hidden_dim: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs, init_state):
        seq_len, input_dim = xs.shape
        Wi = self.param(
            "Wi",
            glorot_normal(in_axis=1, out_axis=0),
            (self.hidden_dim * 3, input_dim),
            self.param_dtype,
        )
        Wh = self.param(
            "Wh",
            orthogonal(column_axis=0),
            (self.hidden_dim * 3, self.hidden_dim),
            self.param_dtype,
        )
        bi = self.param("bi", zeros_init(), (self.hidden_dim * 3,), self.param_dtype)
        bn = self.param("bn", zeros_init(), (self.hidden_dim,), self.param_dtype)

        def _step_fn(h, x):
            igates = jnp.split(Wi @ x + bi, 3)
            hgates = jnp.split(Wh @ h, 3)

            reset = nn.sigmoid(igates[0] + hgates[0])
            update = nn.sigmoid(igates[1] + hgates[1])
            new = nn.tanh(igates[2] + reset * (hgates[2] + bn))
            next_h = (1 - update) * new + update * h

            return next_h, next_h

        xs, init_state, Wi, Wh, bi, bn = promote_dtype(
            xs, init_state, Wi, Wh, bi, bn, dtype=self.dtype
        )
        last_state, all_states = jax.lax.scan(_step_fn, init=init_state, xs=xs)
        return all_states, last_state


class RNNModel(nn.Module):
    hidden_dim: int
    num_layers: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs, init_state):
        # xs: [seq_len, input_dim]
        # init_state: [num_layers, hidden_dim]
        outs, states = [], []
        for layer in range(self.num_layers):
            xs, state = GRU(self.hidden_dim, self.dtype, self.param_dtype)(
                xs, init_state[layer]
            )
            outs.append(xs)
            states.append(state)
        # sum outputs from all layers
        return jnp.array(outs).sum(0), jnp.array(states)


BatchedRNNModel = flax.linen.vmap(
    RNNModel,
    variable_axes={"params": None},
    split_rngs={"params": False},
    axis_name="batch",
)


# ---------------------------------------------------------------------------
# Actor-Critic with MLP observation encoder
# ---------------------------------------------------------------------------


class ActorCriticInput(TypedDict):
    obs: jax.Array  # [batch, seq, obs_dim]
    prev_action: jax.Array  # [batch, seq]
    prev_reward: jax.Array  # [batch, seq]


class ActorCriticMLP(nn.Module):
    """MLP-based actor-critic with GRU for memory.

    Accepts flat 1D observations from MDPInterfaceWrapper instead of the
    grid-specific (H, W, 2) observations used by xland-minigrid's
    ActorCriticRNN.
    """

    num_actions: int
    hidden_dim: int = 256
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 512
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self, inputs: ActorCriticInput, hidden: jax.Array
    ) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        B, S = inputs["obs"].shape[:2]

        obs_encoder = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.relu,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(math.sqrt(2)),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.relu,
            ]
        )
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)

        rnn_core = BatchedRNNModel(
            self.rnn_hidden_dim,
            self.rnn_num_layers,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        actor = nn.Sequential(
            [
                nn.Dense(
                    self.head_hidden_dim,
                    kernel_init=orthogonal(2),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.tanh,
                nn.Dense(
                    self.num_actions,
                    kernel_init=orthogonal(0.01),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
            ]
        )
        critic = nn.Sequential(
            [
                nn.Dense(
                    self.head_hidden_dim,
                    kernel_init=orthogonal(2),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.tanh,
                nn.Dense(
                    1,
                    kernel_init=orthogonal(1.0),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
            ]
        )

        # [batch, seq, hidden_dim]
        obs_emb = obs_encoder(inputs["obs"].astype(jnp.float32))
        act_emb = action_encoder(inputs["prev_action"])

        # [batch, seq, hidden_dim + action_emb_dim + 1]
        out = jnp.concatenate(
            [obs_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1
        )

        out, new_hidden = rnn_core(out, hidden)

        logits = actor(out).astype(jnp.float32)
        dist = distrax.Categorical(logits=logits)
        values = critic(out)

        return dist, jnp.squeeze(values, axis=-1), new_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros(
            (batch_size, self.rnn_num_layers, self.rnn_hidden_dim), dtype=self.dtype
        )
