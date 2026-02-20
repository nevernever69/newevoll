"""Easy task: Pick up the blue pyramid.

Single room (R1, 9x9), one target object (blue pyramid), one distractor (red ball).
Goal: AgentHoldGoal — agent must be holding the blue pyramid.
No rules (no transformations).
"""

import jax.numpy as jnp
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import AgentHoldGoal
from xminigrid.core.rules import EmptyRule
from xminigrid.types import RuleSet


def build_ruleset() -> RuleSet:
    blue_pyramid = TILES_REGISTRY[Tiles.PYRAMID, Colors.BLUE]
    red_ball = TILES_REGISTRY[Tiles.BALL, Colors.RED]

    goal = AgentHoldGoal(tile=blue_pyramid)
    rules = EmptyRule().encode()[None, ...]
    init_tiles = jnp.array([blue_pyramid, red_ball])

    return RuleSet(
        goal=goal.encode(),
        rules=rules,
        init_tiles=init_tiles,
    )
