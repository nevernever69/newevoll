"""Hard task: Pick up blue pyramid (transforms to green ball), place near yellow hex.

Four rooms (R4, 13x13), one rule, two distractors.

Rule: AgentHoldRule — when the agent picks up the blue pyramid, it instantly
      transforms into a green ball in the agent's pocket.
Goal: TileNearGoal — green ball must be adjacent to yellow hex.

Distractors: purple ball and orange square (with a useless TileNearRule that
produces a grey star — a dead end).

The agent must:
1. Find and pick up the blue pyramid (it becomes a green ball in hand)
2. Navigate to the yellow hex (possibly through doors between rooms)
3. Put the green ball down next to the yellow hex (satisfies the goal)
"""

import jax.numpy as jnp
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import TileNearGoal
from xminigrid.core.rules import AgentHoldRule, TileNearRule
from xminigrid.types import RuleSet


def build_ruleset() -> RuleSet:
    blue_pyramid = TILES_REGISTRY[Tiles.PYRAMID, Colors.BLUE]
    green_ball = TILES_REGISTRY[Tiles.BALL, Colors.GREEN]
    yellow_hex = TILES_REGISTRY[Tiles.HEX, Colors.YELLOW]

    # Distractors
    purple_ball = TILES_REGISTRY[Tiles.BALL, Colors.PURPLE]
    orange_square = TILES_REGISTRY[Tiles.SQUARE, Colors.ORANGE]

    # Rule: picking up blue pyramid transforms it into green ball in pocket
    rule = AgentHoldRule(
        tile=blue_pyramid,
        prod_tile=green_ball,
    )

    # Distractor rule: does nothing useful (transforms purple ball into star
    # when near orange square — a dead end that doesn't help the goal)
    distractor_rule = TileNearRule(
        tile_a=purple_ball,
        tile_b=orange_square,
        prod_tile=TILES_REGISTRY[Tiles.STAR, Colors.GREY],
    )

    goal = TileNearGoal(tile_a=green_ball, tile_b=yellow_hex)
    rules = jnp.vstack([rule.encode(), distractor_rule.encode()])
    init_tiles = jnp.array([
        blue_pyramid,
        yellow_hex,
        purple_ball,
        orange_square,
    ])

    return RuleSet(
        goal=goal.encode(),
        rules=rules,
        init_tiles=init_tiles,
    )
