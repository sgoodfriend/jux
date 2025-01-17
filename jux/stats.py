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
    water_in_factories: jax.Array  # int[2]

    @classmethod
    def empty(cls):
        return cls(
            lichen=jnp.zeros((2, ), dtype=jnp.int32),
            light_bots=jnp.zeros((2, ), dtype=jnp.int32),
            heavy_bots=jnp.zeros((2, ), dtype=jnp.int32),
            factories=jnp.zeros((2, ), dtype=jnp.int8),
            water_in_factories=jnp.zeros((2, ), dtype=jnp.int32),
        )

    @classmethod
    def from_state(cls, state: "State"):
        bots = state.units.unit_id != imax(Unit.__annotations__["unit_id"])
        return cls(
            lichen=state.team_lichen_score(),
            light_bots=jnp.logical_and(bots, state.units.unit_type == UnitType.LIGHT).sum(1),
            heavy_bots=jnp.logical_and(bots, state.units.unit_type == UnitType.HEAVY).sum(1),
            factories=state.n_factories + state.teams.factories_to_place,
            # Only count water in factories after all factories placed.
            water_in_factories=jnp.where(
                state.real_env_steps >= 0,
                state.factories.cargo.water.sum(1),
                state.teams.init_water,
            ),
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
    queue_update_failures: jax.Array  # int[2]
    valid_move: jax.Array  # int[2]
    valid_transfer: jax.Array  # int[2]
    valid_pickup: jax.Array  # int[2]
    valid_dig: jax.Array  # int[2]
    valid_self_destruct: jax.Array  # int[2]
    valid_recharge: jax.Array  # int[2]

    @classmethod
    def empty(cls):
        return cls(
            queue_update_success=jnp.zeros((2, ), dtype=jnp.int32),
            queue_update_total=jnp.zeros((2, ), dtype=jnp.int32),
            queue_update_failures=jnp.zeros((2, ), dtype=jnp.int32),
            valid_move=jnp.zeros((2, ), dtype=jnp.int32),
            valid_transfer=jnp.zeros((2, ), dtype=jnp.int32),
            valid_pickup=jnp.zeros((2, ), dtype=jnp.int32),
            valid_dig=jnp.zeros((2, ), dtype=jnp.int32),
            valid_self_destruct=jnp.zeros((2, ), dtype=jnp.int32),
            valid_recharge=jnp.zeros((2, ), dtype=jnp.int32),
        )

    @classmethod
    def epsilon(cls, **kwargs):
        kwargs_not_in_cls = [k for k in kwargs if k not in cls._fields]
        assert not kwargs_not_in_cls, f"{kwargs_not_in_cls} not in {cls.__name__}"
        kwargs = {f: kwargs.get(f, 1e-6) for f in cls._fields}
        return cls(**kwargs)


class TransferStats(NamedTuple):
    to_factory: jax.Array  # int[2]
    to_unit: jax.Array  # int[2]
    to_nothing: jax.Array  # int[2]

    @classmethod
    def empty(cls):
        return cls(
            to_factory=jnp.zeros((2, ), dtype=jnp.int32),
            to_unit=jnp.zeros((2, ), dtype=jnp.int32),
            to_nothing=jnp.zeros((2, ), dtype=jnp.int32),
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
    transfers: TransferStats  # TransferStats[2]

    @classmethod
    def empty(cls):
        return cls(**{_f: _type.empty() for _f, _type in cls.__annotations__.items()})

    @classmethod
    def epsilon(cls, **kwargs):
        kwargs_not_in_cls = [k for k in kwargs if k not in cls._fields]
        assert not kwargs_not_in_cls, f"{kwargs_not_in_cls} not in {cls.__name__}"
        kwargs = {_f: _type.epsilon(**kwargs.get(_f, {})) for _f, _type in cls.__annotations__.items()}
        return cls(**kwargs)
