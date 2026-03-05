import sys
import os
import traceback
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, "mujoco_playground")

try:
    from test_go1_pushrecovery import Go1PushRecovery
    env = Go1PushRecovery()
    print("✓ Environment initialized successfully")

    import jax
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    print(f"✓ Reset successful, state.obs shape: {state.obs['state'].shape}")

    import jax.numpy as jnp
    action = jnp.zeros(12)
    next_state = env.step(state, action)
    print(f"✓ Step successful, reward: {next_state.reward}")

except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()
