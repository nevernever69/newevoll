"""Medium task: Place the yellow pyramid adjacent to the green square.

Single room (R1, 9x9), two target objects (yellow pyramid, green square).
Goal: TileNearGoal — yellow pyramid must be adjacent to green square on the grid.
No rules (no transformations).

The agent must: pick up the yellow pyramid, navigate to the green square,
and put the pyramid down next to it.
"""

import jax.numpy as jnp
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import TileNearGoal
from xminigrid.core.rules import EmptyRule
from xminigrid.types import RuleSet


def build_ruleset() -> RuleSet:
    yellow_pyramid = TILES_REGISTRY[Tiles.PYRAMID, Colors.YELLOW]
    green_square = TILES_REGISTRY[Tiles.SQUARE, Colors.GREEN]

    goal = TileNearGoal(tile_a=yellow_pyramid, tile_b=green_square)
    rules = EmptyRule().encode()[None, ...]
    init_tiles = jnp.array([yellow_pyramid, green_square])

    return RuleSet(
        goal=goal.encode(),
        rules=rules,
        init_tiles=init_tiles,
    )
