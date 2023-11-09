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
    friendly_kills: jax.Array  # int[2]
    opponent_kills: jax.Array  # int[2]

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
            friendly_kills=jnp.zeros((2, ), dtype=jnp.int32),
            opponent_kills=jnp.zeros((2, ), dtype=jnp.int32),
        )

    @classmethod
    def epsilon(cls, **kwargs):
        kwargs_not_in_cls = [k for k in kwargs if k not in cls._fields]
        assert not kwargs_not_in_cls, f"{kwargs_not_in_cls} not in {cls.__name__}"
        kwargs = {f: kwargs.get(f, 1e-6) for f in cls._fields}
        return cls(**kwargs)


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
    def from_state(cls, state: "State"):
        bots = state.units.unit_id != imax(Unit.__annotations__["unit_id"])
        return cls(
            lichen=state.team_lichen_score(),
            light_bots=jnp.logical_and(bots, state.units.unit_type == UnitType.LIGHT).sum(1),
            heavy_bots=jnp.logical_and(bots, state.units.unit_type == UnitType.HEAVY).sum(1),
            factories=state.n_factories,
        )

    @classmethod
    def epsilon(cls, **kwargs):
        kwargs_not_in_cls = [k for k in kwargs if k not in cls._fields]
        assert not kwargs_not_in_cls, f"{kwargs_not_in_cls} not in {cls.__name__}"
        kwargs = {f: kwargs.get(f, 1e-6) for f in cls._fields}
        return cls(**kwargs)


class ActionStats(NamedTuple):
    queue_update_success: jax.Array  # int[2]
    queue_update_total: jax.Array  # int[2]

    @classmethod
    def empty(cls):
        return cls(
            queue_update_success=jnp.zeros((2, ), dtype=jnp.int32),
            queue_update_total=jnp.zeros((2, ), dtype=jnp.int32),
        )

    @classmethod
    def epsilon(cls, **kwargs):
        kwargs_not_in_cls = [k for k in kwargs if k not in cls._fields]
        assert not kwargs_not_in_cls, f"{kwargs_not_in_cls} not in {cls.__name__}"
        kwargs = {f: kwargs.get(f, 1e-6) for f in cls._fields}
        return cls(**kwargs)


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

    @classmethod
    def epsilon(cls, **kwargs):
        kwargs_not_in_cls = [k for k in kwargs if k not in cls._fields]
        assert not kwargs_not_in_cls, f"{kwargs_not_in_cls} not in {cls.__name__}"
        kwargs = {_f: _type.epsilon(**kwargs.get(_f, {})) for _f, _type in cls._field_types.items()}
        return cls(**kwargs)
