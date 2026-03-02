# PandaTracking Library Context

You are writing JAX-compatible Python functions (`get_observation`, `compute_reward`)
that operate on MuJoCo Playground environment states for a Franka Emika Panda robot arm.

---

## 1. Task Description

The Panda arm must **track a moving 3D target** (Lissajous trajectory) with its end-effector
as precisely as possible. Single-phase task — pure tracking, no grasping or manipulation.

**Success**: Mean end-effector-to-target tracking error < 0.02m (2cm) over the episode.

**Episode termination**: NaN in state. Max 500 steps (10s).

---

## 2. State Object

| Field | Type | Description |
|---|---|---|
| `state.data` | `mjx.Data` | Full MuJoCo simulation state |
| `state.data.qpos` | `jax.Array (16,)` | See qpos layout below |
| `state.data.qvel` | `jax.Array (15,)` | See qvel layout below |
| `state.data.ctrl` | `jax.Array (8,)` | Current control targets (7 arm + 1 gripper) |
| `state.obs` | `jax.Array` | Environment's default 42-dim observation |
| `state.reward` | `jax.Array ()` | Current reward scalar |
| `state.done` | `jax.Array ()` | Termination flag (0.0 or 1.0) |
| `state.info` | `dict` | Auxiliary information (see below) |
| `state.metrics` | `dict` | Logged metrics |

---

## 3. qpos / qvel Layout

**qpos (16 dimensions):**

| Indices | Dim | Content |
|---------|-----|---------|
| 0–6 | 7 | Arm joint angles |
| 7 | 1 | Left finger position (slide, 0.0=closed, 0.04=open) |
| 8 | 1 | Right finger position (mirrors left finger) |
| 9–11 | 3 | Box position (x, y, z) — present in scene but unused |

**qvel (15 dimensions):**

| Indices | Dim | Content |
|---------|-----|---------|
| 0–6 | 7 | Arm joint velocities |
| 7–8 | 2 | Finger velocities |
| 9–11 | 3 | Box translational velocity (unused) |
| 12–14 | 3 | Box rotational velocity (unused) |

---

## 4. state.info Fields (Pre-computed for Discovery)

| Field | Shape | Description |
|---|---|---|
| `state.info["target_pos"]` | `(3,)` | Current Lissajous target position (x, y, z) |
| `state.info["target_vel"]` | `(3,)` | Current target velocity vector (analytical derivative) |
| `state.info["gripper_pos"]` | `(3,)` | End-effector (gripper site) position |
| `state.info["gripper_target_dist"]` | `()` | Euclidean distance between gripper and target |
| `state.info["step_count"]` | `()` | Current timestep in episode (0 to 499) |
| `state.info["prev_ctrl"]` | `(8,)` | Previous control targets (7 arm + 1 gripper) |
| `state.info["traj_params"]` | `dict` | Lissajous params: `freq_x`, `freq_y`, `freq_z`, `phase_x`, `phase_y` |
| `state.info["cumulative_tracking_error"]` | `()` | Sum of gripper-to-target distance over episode |
| `state.info["tracking_steps"]` | `()` | Number of steps accumulated (denominator for mean tracking error) |

---

## 5. Action Space (8 actuators)

Actions are **delta joint positions** scaled by `action_scale=0.04` and added to current ctrl:
`ctrl = clip(state.data.ctrl + action * 0.04, lower_limits, upper_limits)`

| Index | Joint | Range |
|---|---|---|
| 0 | joint1 (shoulder) | [-2.897, 2.897] |
| 1 | joint2 (shoulder) | [-1.763, 1.763] |
| 2 | joint3 (elbow) | [-2.897, 2.897] |
| 3 | joint4 (elbow) | [-3.072, -0.070] |
| 4 | joint5 (wrist) | [-2.897, 2.897] |
| 5 | joint6 (wrist) | [-0.018, 3.753] |
| 6 | joint7 (wrist) | [-2.897, 2.897] |
| 7 | **gripper** | **[0, 0.04]** — not used for this task but present |

---

## 6. Physical Constants

**Target trajectory** (Lissajous curve):
```
x = 0.45 + 0.10 * sin(0.35 * freq_x * t + phase_x)
y = 0.00 + 0.10 * sin(0.35 * freq_y * t + phase_y)
z = 0.18 + 0.07 * sin(0.35 * freq_z * t)
```
- Workspace: x ∈ [0.35, 0.55], y ∈ [-0.10, 0.10], z ∈ [0.11, 0.25]
- Frequencies: freq_x, freq_y ∈ [1.0, 3.0]; freq_z ∈ [0.8, 2.0] (randomized per episode)
- Phases: phase_x, phase_y ∈ [0, 2π] (randomized per episode)

**Episode**: 500 steps, ctrl_dt=0.02s (10 seconds total).

---

## 7. JAX Constraints

Your code must be **JAX-traceable** (`jax.jit` / `jax.vmap` compatible).

**Do:**
```python
import jax
import jax.numpy as jnp

dist = jnp.linalg.norm(state.info["gripper_pos"] - state.info["target_pos"])
reward = 1.0 - jnp.tanh(5.0 * dist)
obs = jnp.concatenate([feature1, feature2, feature3])
```

**Do NOT:**
```python
if dist > 0.05:                        # BAD — Python if on JAX array
    reward = 1.0
import numpy as np                      # BAD — must use jax.numpy
for i in range(len(qpos)):              # BAD — use vectorized jnp ops
```

Only `import jax` and `import jax.numpy as jnp` are allowed.

---

## 8. Function Signatures

```python
def get_observation(state) -> jnp.ndarray:
    """Return a 1D float array of observations."""
    # state is the MuJoCo State object
    # Must return shape (obs_dim,) where obs_dim <= 512
    ...

def compute_reward(state, action, next_state) -> jnp.ndarray:
    """Return a scalar reward."""
    # state: pre-step state, action: (8,) continuous, next_state: post-step state
    # Must return a scalar float
    ...
```

Note: `action` is a continuous array of shape `(8,)`, not a discrete integer.
