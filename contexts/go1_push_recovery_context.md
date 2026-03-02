# Go1PushRecovery Library Context

You are writing JAX-compatible Python functions (`get_observation`, `compute_reward`)
that operate on MuJoCo Playground environment states for a Unitree Go1 quadruped robot.

---

## 1. Task Description

The Go1 quadruped **stands at the origin**. Random horizontal force impulses are applied
to the torso at regular intervals (~every 1.5s, lasting ~0.1s, 150-400N). The robot must:

- **Stay upright** (don't fall over)
- **Return to origin** after each push
- **Maintain heading** (face the same direction)

**Success**: Survived full episode (500 steps) AND average position error < 10cm from origin.

**Termination**: Episode ends early if `upvector[-1] < 0.3` (body tilted > ~72° from vertical).

---

## 2. State Object

The `state` argument is a MuJoCo Playground `State` dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `state.data` | `mjx.Data` | Full MuJoCo simulation state |
| `state.data.qpos` | `jax.Array (19,)` | Positions: [x,y,z, qw,qx,qy,qz, 12×joint_angles] |
| `state.data.qvel` | `jax.Array (18,)` | Velocities: [vx,vy,vz, wx,wy,wz, 12×joint_vels] |
| `state.data.ctrl` | `jax.Array (12,)` | Current motor control targets |
| `state.data.actuator_force` | `jax.Array (12,)` | Current actuator forces |
| `state.obs` | `dict` | `{"state": jax.Array (52,)}` — environment's default observation |
| `state.reward` | `jax.Array ()` | Current reward scalar |
| `state.done` | `jax.Array ()` | Termination flag (0.0 or 1.0) |
| `state.info` | `dict` | Auxiliary information (see below) |
| `state.metrics` | `dict` | Logged metrics |

---

## 3. state.info Fields (Pre-computed for Discovery)

| Field | Shape | Description |
|---|---|---|
| `state.info["gyro"]` | `(3,)` | Angular velocity from IMU sensor |
| `state.info["gravity"]` | `(3,)` | Gravity vector in body frame |
| `state.info["local_linvel"]` | `(3,)` | Linear velocity in body frame |
| `state.info["upvector"]` | `(3,)` | Body up-vector in world frame; `upvector[-1] < 0.3` → fallen |
| `state.info["pos_xy"]` | `(2,)` | XY offset from origin (where robot should return to) |
| `state.info["height"]` | `()` | COM height (z-coordinate); standing height is ~0.278m |
| `state.info["heading"]` | `()` | Current yaw angle (radians); target heading is ~0 |
| `state.info["push_force"]` | `(3,)` | Current push force vector (raw Newtons, NOT normalized). Mostly horizontal with slight upward component. Zero between pushes. |
| `state.info["last_act"]` | `(12,)` | Previous action |
| `state.info["step_count"]` | `()` | Current timestep in episode (0 to 499) |
| `state.info["push_count"]` | `()` | Number of pushes so far |
| `state.info["cum_pos_error"]` | `()` | Cumulative position error (used for success check) |

---

## 4. Action Space

**12 continuous actions** (joint position offsets for all 4 legs × 3 joints each).
Actions are scaled by `action_scale=0.5` and added to the default standing pose:
`motor_targets = default_pose + action * 0.5`

Joint order (12 total):

| Index | Joint | Leg | Type | Range (rad) |
|---|---|---|---|---|
| 0 | FR_hip_joint | Front-Right | abduction | [-0.863, 0.863] |
| 1 | FR_thigh_joint | Front-Right | hip | [-0.686, 4.501] |
| 2 | FR_calf_joint | Front-Right | knee | [-2.818, -0.888] |
| 3 | FL_hip_joint | Front-Left | abduction | [-0.863, 0.863] |
| 4 | FL_thigh_joint | Front-Left | hip | [-0.686, 4.501] |
| 5 | FL_calf_joint | Front-Left | knee | [-2.818, -0.888] |
| 6 | RR_hip_joint | Rear-Right | abduction | [-0.863, 0.863] |
| 7 | RR_thigh_joint | Rear-Right | hip | [-0.686, 4.501] |
| 8 | RR_calf_joint | Rear-Right | knee | [-2.818, -0.888] |
| 9 | RL_hip_joint | Rear-Left | abduction | [-0.863, 0.863] |
| 10 | RL_thigh_joint | Rear-Left | hip | [-0.686, 4.501] |
| 11 | RL_calf_joint | Rear-Left | knee | [-2.818, -0.888] |

---

## 5. Physical Constants

**Default standing pose** (joint angles at home keyframe):
```python
DEFAULT_POSE = jnp.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8])
# Per-leg: [abduction, hip, knee] for FR, FL, RR, RL
```

**Standing height**: 0.278 m (COM z-coordinate when upright)

**Push timing**:
- Pushes occur every ~75 steps (~1.5s), each lasting ~5 steps (~0.1s)
- Push direction: random angle in horizontal plane with slight upward component
- `push_force` in `state.info` is the force vector during a push; zero between pushes. You can detect active pushes via `jnp.linalg.norm(push_force) > 0`

---

## 6. JAX Constraints

Your code must be **JAX-traceable** (`jax.jit` / `jax.vmap` compatible).

**Do:**
```python
import jax
import jax.numpy as jnp

dist = jnp.linalg.norm(state.info["pos_xy"])
reward = 1.0 - jnp.tanh(5.0 * dist)
obs = jnp.concatenate([state.info["gyro"], state.info["gravity"], ...])
```

**Do NOT:**
```python
if state.info["upvector"][-1] < 0.3:    # BAD — Python if on JAX array
    reward = -1.0
import numpy as np                        # BAD — must use jax.numpy
for joint in range(12):                   # BAD — use vectorized jnp ops
```

Only `import jax` and `import jax.numpy as jnp` are allowed.

---

## 7. Function Signatures

```python
def get_observation(state) -> jnp.ndarray:
    """Return a 1D float array of observations."""
    # state is the MuJoCo State object
    # Must return shape (obs_dim,) where obs_dim <= 512
    ...

def compute_reward(state, action, next_state) -> jnp.ndarray:
    """Return a scalar reward."""
    # state: pre-step state, action: (12,) continuous, next_state: post-step state
    # Must return a scalar float
    ...
```

Note: `action` is a continuous array of shape `(12,)`, not a discrete integer.
