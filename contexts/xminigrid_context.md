# XLand-MiniGrid Library Context

You are writing JAX-compatible Python functions (`get_observation`, `compute_reward`)
that operate on xland-minigrid environment states.

---

## 1. State Object

| Field | Type / Shape | Description |
|---|---|---|
| `state.grid` | `jax.Array (H, W, 2)` uint8 | Grid of tiles. `grid[y, x, 0]` = tile_id, `grid[y, x, 1]` = color_id. |
| `state.agent.position` | `jax.Array (2,)` uint8 | Agent position as `[row, col]` (0-indexed). |
| `state.agent.direction` | `jax.Array ()` uint8 | Facing direction: 0=Up, 1=Right, 2=Down, 3=Left. |
| `state.agent.pocket` | `jax.Array (2,)` uint8 | Held item `[tile_id, color_id]`. `[0, 0]` = empty. |
| `state.step_num` | `jax.Array ()` uint8 | Current timestep. |

Coordinate system: `grid[y, x, layer]`. y increases downward, x increases rightward.

---

## 2. Tile Constants

| Name | ID | Notes |
|---|---|---|
| EMPTY | 0 | Out-of-bounds / padding |
| FLOOR | 1 | Walkable empty tile |
| WALL | 2 | Impassable barrier |
| BALL | 3 | Pickable object |
| SQUARE | 4 | Pickable object |
| PYRAMID | 5 | Pickable object |
| GOAL | 6 | Walkable goal marker |
| KEY | 7 | Pickable; unlocks matching-color locked door |
| DOOR_LOCKED | 8 | Impassable until unlocked with matching key |
| DOOR_CLOSED | 9 | Impassable until toggled open |
| DOOR_OPEN | 10 | Walkable open door |
| HEX | 11 | Pickable object |
| STAR | 12 | Pickable object |

Pickable: BALL(3), SQUARE(4), PYRAMID(5), KEY(7), HEX(11), STAR(12).

---

## 3. Color Constants

| Name | ID | Name | ID |
|---|---|---|---|
| EMPTY | 0 | GREY | 6 |
| RED | 1 | BLACK | 7 |
| GREEN | 2 | ORANGE | 8 |
| BLUE | 3 | WHITE | 9 |
| PURPLE | 4 | BROWN | 10 |
| YELLOW | 5 | PINK | 11 |

Floor tiles are BLACK (color_id=7).

---

## 4. Action Space

6 discrete actions: FORWARD(0), TURN_RIGHT(1), TURN_LEFT(2), PICKUP(3), PUTDOWN(4), TOGGLE(5).

- PICKUP: picks up pickable object in front (if pocket empty)
- PUTDOWN: places held object on floor in front
- TOGGLE: unlock locked door (with matching key), open/close doors

---

## 5. Direction Vectors

| Direction | ID | Delta (dy, dx) |
|---|---|---|
| Up | 0 | (−1, 0) |
| Right | 1 | (0, +1) |
| Down | 2 | (+1, 0) |
| Left | 3 | (0, −1) |

```python
DIRECTIONS = jnp.array([(-1,0), (0,1), (1,0), (0,-1)])
front_pos = state.agent.position + DIRECTIONS[state.agent.direction]
```

---

## 6. JAX Constraints

Your code must be **JAX-traceable** (`jax.jit` / `jax.vmap` compatible).

**Do:**
```python
import jax
import jax.numpy as jnp

reward = jax.lax.select(condition, val_true, val_false)  # conditional
mask = (state.grid[:, :, 0] == 5)                        # boolean mask
dir_oh = jax.nn.one_hot(state.agent.direction, 4)        # one-hot
obs = jnp.concatenate([f1, f2, f3])                      # concat
```

**Do NOT:**
```python
if state.agent.position[0] > 3:   # BAD — Python if on JAX array
    reward = 1.0
import numpy as np                 # BAD — must use jax.numpy
for i in range(H):                 # BAD — use vectorized jnp ops
    ...
```

Only `import jax` and `import jax.numpy as jnp` are allowed.

---

## 7. Useful Patterns

```python
# Manhattan distance
dist = jnp.abs(pos1[0] - pos2[0]) + jnp.abs(pos1[1] - pos2[1])

# Find tile positions via meshgrid
H, W = state.grid.shape[:2]
ys, xs = jnp.arange(H), jnp.arange(W)
yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
mask = (state.grid[:, :, 0] == TILE_ID) & (state.grid[:, :, 1] == COLOR_ID)
count = jnp.sum(mask)
mean_y = jnp.sum(yy * mask) / jnp.maximum(count, 1)
mean_x = jnp.sum(xx * mask) / jnp.maximum(count, 1)

# Pocket checks
pocket_empty = (state.agent.pocket[0] == 0)
holding_pyramid = (state.agent.pocket[0] == 5)

# Front tile
DIRECTIONS = jnp.array([(-1,0), (0,1), (1,0), (0,-1)])
dy, dx = DIRECTIONS[state.agent.direction]
front_y = jnp.clip(state.agent.position[0] + dy, 0, H - 1)
front_x = jnp.clip(state.agent.position[1] + dx, 0, W - 1)
front_tile_id = state.grid[front_y, front_x, 0]

# Normalize
norm_y = state.agent.position[0].astype(jnp.float32) / (H - 1)
```
