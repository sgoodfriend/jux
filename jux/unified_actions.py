from typing import NamedTuple

from jax import Array

from jux.actions import JuxAction


class FactoryPlacementAction(NamedTuple):
    spawn: Array  # int[2, 2]
    water: Array  # int[2]
    metal: Array  # int[2]


class UnifiedAction(NamedTuple):
    factory_placement_action: FactoryPlacementAction  # FactoryPlacementAction[2]
    late_game_action: JuxAction  # JuxAction[2]
