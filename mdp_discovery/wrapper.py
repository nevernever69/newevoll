"""MDPInterfaceWrapper — plugs LLM-discovered observation/reward functions
into an xland-minigrid environment."""

from __future__ import annotations

from typing import Callable

from xminigrid.wrappers import Wrapper


class MDPInterfaceWrapper(Wrapper):
    """Replaces the default observation and reward with MDP interface outputs.

    Wrapper stacking order (important!):
        env = MDPInterfaceWrapper(env, get_obs_fn, compute_reward_fn)  # first
        env = GymAutoResetWrapper(env)                                  # second

    This ensures compute_reward sees the true terminal state on episode
    boundaries, before GymAutoResetWrapper replaces it with the reset state.
    """

    def __init__(
        self,
        env,
        get_observation_fn: Callable,
        compute_reward_fn: Callable,
    ):
        super().__init__(env)
        self._get_observation = get_observation_fn
        self._compute_reward = compute_reward_fn

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        new_obs = self._get_observation(timestep.state)
        return timestep.replace(observation=new_obs)

    def step(self, params, timestep, action):
        prev_state = timestep.state
        new_timestep = self._env.step(params, timestep, action)
        new_obs = self._get_observation(new_timestep.state)
        custom_reward = self._compute_reward(prev_state, action, new_timestep.state)
        return new_timestep.replace(observation=new_obs, reward=custom_reward)
