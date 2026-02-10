# XLand-MiniGrid Library Context Document

You are writing JAX-compatible Python functions that operate on xland-minigrid
environment states. This document describes every field, constant, and pattern
you need.

---

## 1. State Object

The environment `state` passed to your functions is a Flax PyTreeNode with these
fields:

| Field | Type / Shape | Description |
|---|---|---|
| `state.grid` | `jax.Array (H, W, 2)` uint8 | Grid of tiles. `grid[y, x, 0]` = tile_id, `grid[y, x, 1]` = color_id. |
| `state.agent.position` | `jax.Array (2,)` uint8 | Agent position as `[row, col]` (0-indexed, row-major). |
| `state.agent.direction` | `jax.Array ()` uint8 | Agent facing direction: 0=Up, 1=Right, 2=Down, 3=Left. |
| `state.agent.pocket` | `jax.Array (2,)` uint8 | Item held by agent as `[tile_id, color_id]`. `[0, 0]` when empty. |
| `state.goal_encoding` | `jax.Array (5,)` uint8 | Encoded goal specification (see §6). |
| `state.rule_encoding` | `jax.Array (N, 7)` uint8 | Encoded rules (see §7). N = max rules for the environment. |
| `state.step_num` | `jax.Array ()` uint8 | Current timestep within the episode. |
| `state.key` | `jax.Array` | PRNG key (do not use in observation/reward). |

Grid sizes depend on the environment. Common sizes: 6×6, 9×9, 13×13.

---

## 2. Tile Constants

There are 13 tile types (stored as uint8 in layer 0 of the grid):

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

**Tile categories:**
- Walkable: FLOOR (1), GOAL (6), DOOR_OPEN (10)
- Pickable: BALL (3), SQUARE (4), PYRAMID (5), KEY (7), HEX (11), STAR (12)
- Blocks line-of-sight: WALL (2), DOOR_LOCKED (8), DOOR_CLOSED (9)

---

## 3. Color Constants

There are 12 colors (stored as uint8 in layer 1 of the grid):

| Name | ID |
|---|---|
| EMPTY | 0 |
| RED | 1 |
| GREEN | 2 |
| BLUE | 3 |
| PURPLE | 4 |
| YELLOW | 5 |
| GREY | 6 |
| BLACK | 7 |
| ORANGE | 8 |
| WHITE | 9 |
| BROWN | 10 |
| PINK | 11 |

Floor tiles are typically BLACK (color_id=7). Colors matter for key/door
matching and tile identity in goals/rules.

---

## 4. Action Space

6 discrete actions (integer 0–5):

| Action | ID | Effect |
|---|---|---|
| FORWARD | 0 | Move one cell in facing direction (only if target is walkable). |
| TURN_RIGHT | 1 | Rotate clockwise: direction = (direction + 1) % 4. |
| TURN_LEFT | 2 | Rotate counter-clockwise: direction = (direction − 1) % 4. |
| PICKUP | 3 | Pick up object in front of agent (if pickable and pocket empty). |
| PUTDOWN | 4 | Place held object in front of agent (if target is floor and pocket not empty). |
| TOGGLE | 5 | Toggle door in front: locked+matching key→open, closed→open, open→closed. |

Agent cannot move through walls, locked doors, or closed doors.

---

## 5. Direction Vectors

| Direction | ID | Delta (dy, dx) |
|---|---|---|
| Up | 0 | (−1, 0) |
| Right | 1 | (0, +1) |
| Down | 2 | (+1, 0) |
| Left | 3 | (0, −1) |

The position in front of the agent is:
```python
front_pos = state.agent.position + DIRECTIONS[state.agent.direction]
# where DIRECTIONS = jnp.array([(-1,0), (0,1), (1,0), (0,-1)])
```

---

## 6. Goal Encoding

`state.goal_encoding` is a `(5,)` uint8 array. The first element is the goal
type ID. There are 15 goal types:

| ID | Goal Type | Encoding | Condition |
|---|---|---|---|
| 0 | EmptyGoal | `[0,0,0,0,0]` | Always false (padding). |
| 1 | AgentHoldGoal | `[1,tile,color,0,0]` | Agent picks up a specific tile. |
| 2 | AgentOnTileGoal | `[2,tile,color,0,0]` | Agent walks onto a specific tile. |
| 3 | AgentNearGoal | `[3,tile,color,0,0]` | Agent moves next to a specific tile. |
| 4 | TileNearGoal | `[4,tA,cA,tB,cB]` | Two specific tiles become adjacent (after put_down). |
| 5 | TileOnPositionGoal | `[5,tile,color,y,x]` | A specific tile is at position (y,x). |
| 6 | AgentOnPositionGoal | `[6,y,x,0,0]` | Agent reaches position (y,x). |
| 7–10 | TileNearDirGoal | `[id,tA,cA,tB,cB]` | Two tiles adjacent in specific direction (Up/Right/Down/Left). |
| 11–14 | AgentNearDirGoal | `[id,tile,color,0,0]` | Agent adjacent to tile in specific direction. |

Goals are checked after every step. When the goal is satisfied, the episode
terminates.

---

## 7. Rule Encoding

`state.rule_encoding` is an `(N, 7)` uint8 array. Each row encodes one rule.
Rules modify the grid/agent state after each action. The first element of each
row is the rule type ID. There are 12 rule types.

| ID | Rule Type | Encoding | Effect |
|---|---|---|---|
| 0 | EmptyRule | `[0,0,0,0,0,0,0]` | No-op (padding). |
| 1 | AgentHoldRule | `[1,t,c,pt,pc,0,0]` | On pickup: if holding tile (t,c), transform pocket to (pt,pc). |
| 2 | AgentNearRule | `[2,t,c,pt,pc,0,0]` | On move: neighbors matching (t,c) become (pt,pc). |
| 3 | TileNearRule | `[3,tA,cA,tB,cB,pt,pc]` | On putdown: if tiles (A,B) adjacent, transform to (pt,pc). |
| 4–7 | TileNearDirRule | `[id,tA,cA,tB,cB,pt,pc]` | Directional tile-near variants (Up/Right/Down/Left). |
| 8–11 | AgentNearDirRule | `[id,t,c,pt,pc,0,0]` | Directional agent-near variants (Up/Right/Down/Left). |

Rules are applied **sequentially** (via `jax.lax.scan`) after the action and
before the goal check.

---

## 8. Step Execution Order

Each environment step executes in this order:

1. **Execute action** → updates grid and agent (move, turn, pick up, etc.)
2. **Apply rules** → all rules checked and applied sequentially
3. **Increment step_num**
4. **Check goal** → determines if episode terminates
5. **Check truncation** → step_num == max_steps
6. **Compute default reward** → 1.0 − 0.9 × (step_num / max_steps) if goal reached, else 0.0
7. **Compute observation** → transparent field-of-view

**Note:** When using the MDP interface wrapper, steps 6 and 7 are replaced by
your `compute_reward` and `get_observation` functions. Steps 1–5 still happen
inside the environment.

---

## 9. Coordinate System

- Grid uses **(row, col)** indexing: `grid[y, x, layer]`.
- `y` increases downward (row 0 is the top).
- `x` increases rightward (col 0 is the left).
- Agent position is `[y, x]`.
- Grid shape is `(height, width, 2)`.

---

## 10. JAX Constraints

Your code must be **JAX-traceable** (compatible with `jax.jit` and `jax.vmap`):

### Do:
```python
import jax
import jax.numpy as jnp

# Arithmetic on arrays
dist = jnp.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# Conditional values
reward = jax.lax.select(condition, value_if_true, value_if_false)

# Boolean masks
wall_mask = (state.grid[:, :, 0] == 2)  # True where walls are
floor_count = jnp.sum(state.grid[:, :, 0] == 1)

# Array slicing and indexing
tile_at_pos = state.grid[y, x]
tile_id = state.grid[y, x, 0]

# Functional array updates
new_grid = state.grid.at[y, x].set(jnp.array([1, 7]))  # set to FLOOR, BLACK

# One-hot encoding
dir_onehot = jax.nn.one_hot(state.agent.direction, 4)

# Concatenation
obs = jnp.concatenate([feature1, feature2, feature3])

# Flattening
flat = state.grid.reshape(-1).astype(jnp.float32)

# Clipping
pos = jnp.clip(pos, 0, max_val)

# Dynamic indexing
direction_vec = jax.lax.dynamic_index_in_dim(DIRECTIONS, state.agent.direction, keepdims=False)
```

### Do NOT:
```python
# Python if/else on JAX arrays (breaks tracing)
if state.agent.position[0] > 3:  # BAD
    reward = 1.0

# Python for loops over JAX array elements
for i in range(state.grid.shape[0]):  # BAD — use jnp operations instead
    ...

# NumPy (must use jax.numpy)
import numpy as np  # BAD
np.array(...)       # BAD

# In-place mutation
state.grid[y, x] = value  # BAD — use .at[].set()

# Python lists of JAX arrays in hot loops
results = []  # BAD in traced code
for x in xs:
    results.append(f(x))
```

---

## 11. Available Imports

Your MDP interface file may only use:

```python
import jax
import jax.numpy as jnp
```

No other imports are available. All computation must use JAX primitives.

---

## 12. Useful Patterns

### Distance calculations
```python
# Manhattan distance between two positions
dist = jnp.abs(pos1[0] - pos2[0]) + jnp.abs(pos1[1] - pos2[1])

# Euclidean distance
dist = jnp.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```

### Counting tiles of a given type
```python
# Count all GOAL tiles
goal_count = jnp.sum(state.grid[:, :, 0] == 6)

# Count all KEY tiles
key_count = jnp.sum(state.grid[:, :, 0] == 7)
```

### Finding tile positions
```python
# Boolean mask of where GOAL tiles are
goal_mask = (state.grid[:, :, 0] == 6)

# Get coordinates using meshgrid
H, W = state.grid.shape[:2]
ys = jnp.arange(H)
xs = jnp.arange(W)
yy, xx = jnp.meshgrid(ys, xs, indexing='ij')

# Mean position of all goal tiles (if any)
goal_y = jnp.sum(yy * goal_mask) / jnp.maximum(jnp.sum(goal_mask), 1)
goal_x = jnp.sum(xx * goal_mask) / jnp.maximum(jnp.sum(goal_mask), 1)
```

### Checking agent's pocket
```python
# Is pocket empty?
pocket_empty = (state.agent.pocket[0] == 0)

# Is agent holding a key?
holding_key = (state.agent.pocket[0] == 7)

# What color key?
key_color = state.agent.pocket[1]
```

### Local neighborhood
```python
pos = state.agent.position
y, x = pos[0], pos[1]
H, W = state.grid.shape[:2]

# Tile directly in front of agent
DIRECTIONS = jnp.array([(-1,0), (0,1), (1,0), (0,-1)])
dy, dx = DIRECTIONS[state.agent.direction]
front_y = jnp.clip(y + dy, 0, H - 1)
front_x = jnp.clip(x + dx, 0, W - 1)
front_tile = state.grid[front_y, front_x, 0]
```

### Normalizing features
```python
# Normalize position to [0, 1]
H, W = state.grid.shape[:2]
norm_y = state.agent.position[0].astype(jnp.float32) / (H - 1)
norm_x = state.agent.position[1].astype(jnp.float32) / (W - 1)

# Normalize step count
norm_step = state.step_num.astype(jnp.float32) / 100.0  # or max_steps if known
```

---

## 13. Registered Environments

| Environment ID | Grid Size | Description |
|---|---|---|
| MiniGrid-Empty-6x6 | 6×6 | Empty room, agent must reach goal |
| MiniGrid-Empty-8x8 | 8×8 | Larger empty room |
| MiniGrid-DoorKey-5x5 | 5×5 | Pick up key, open door, reach goal |
| MiniGrid-DoorKey-6x6 | 6×6 | Larger door-key puzzle |
| MiniGrid-FourRooms | 19×19 | Four connected rooms |
| MiniGrid-Unlock | 11×11 | Find key and unlock door |
| XLand-MiniGrid-R1-9x9 | 9×9 | Single procedural room |
| XLand-MiniGrid-R2-13x13 | 13×13 | Two procedural rooms |
| XLand-MiniGrid-R4-13x13 | 13×13 | Four procedural rooms |

XLand environments support custom rulesets (goals + rules + initial tiles)
loaded from benchmarks.
