from typing import NamedTuple

import jax
import jax.numpy as jnp

from jux.unit import Unit, UnitType
from jux.utils import imax


class GenerationStats(NamedTuple):
    ice: jax.Array  # int[2]
    ore: jax.Array  # int[2]
    water: jax.Array  # int[2]
    metal: jax.Array  # int[2]
    power: jax.Array  # int[2]
    light_bots: jax.Array  # int[2]
    heavy_bots: jax.Array  # int[2]

    @classmethod
    def empty(cls):
        return cls(
            ice=jnp.zeros((2, ), dtype=jnp.int32),
            ore=jnp.zeros((2, ), dtype=jnp.int32),
            water=jnp.zeros((2, ), dtype=jnp.int32),
            metal=jnp.zeros((2, ), dtype=jnp.int32),
            power=jnp.zeros((2, ), dtype=jnp.int32),
            light_bots=jnp.zeros((2, ), dtype=jnp.int32),
            heavy_bots=jnp.zeros((2, ), dtype=jnp.int32),
        )


class ResourceStats(NamedTuple):
    lichen: jax.Array  # int[2]
    light_bots: jax.Array  # int[2]
    heavy_bots: jax.Array  # int[2]
    factories: jax.Array  # int[2]

    @classmethod
    def empty(cls):
        return cls(
            lichen=jnp.zeros((2, ), dtype=jnp.int32),
            light_bots=jnp.zeros((2, ), dtype=jnp.int32),
            heavy_bots=jnp.zeros((2, ), dtype=jnp.int32),
            factories=jnp.zeros((2, ), dtype=jnp.int8),
        )

    @classmethod
    def from_state(cls, state: 'State'):
        bots = state.units.unit_id != imax(Unit.__annotations__['unit_id'])
        return cls(
            lichen=state.team_lichen_score(),
            light_bots=jnp.logical_and(bots, state.units.unit_type == UnitType.LIGHT).sum(1),
            heavy_bots=jnp.logical_and(bots, state.units.unit_type == UnitType.HEAVY).sum(1),
            factories=state.n_factories,
        )


class ActionStats(NamedTuple):
    queue_update_success: jax.Array  # int[2]
    queue_update_total: jax.Array  # int[2]

    @classmethod
    def empty(cls):
        return cls(
            queue_update_success=jnp.zeros((2, ), dtype=jnp.int32),
            queue_update_total=jnp.zeros((2, ), dtype=jnp.int32),
        )


class Stats(NamedTuple):
    generation: GenerationStats  # GenerationStats[2]
    resources: ResourceStats  # ResourceStats[2]
    actions: ActionStats  # ActionStats[2]

    @classmethod
    def empty(cls):
        return cls(
            generation=GenerationStats.empty(),
            resources=ResourceStats.empty(),
            actions=ActionStats.empty(),
        )
