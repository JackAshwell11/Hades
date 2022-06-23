"""
Stores various constants related to consumables and the dataclasses used for
constructing the consumables.
"""
from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import arcade

__all__ = (
    "InstantEffectType",
    "StatusEffectType",
    "ConsumableData",
    "InstantData",
    "StatusEffectData",
    "HEALTH_POTION_INCREASE",
    "ARMOUR_POTION_INCREASE",
    "HEALTH_BOOST_POTION_INCREASE",
    "HEALTH_BOOST_POTION_DURATION",
    "ARMOUR_BOOST_POTION_INCREASE",
    "ARMOUR_BOOST_POTION_DURATION",
    "SPEED_BOOST_POTION_INCREASE",
    "SPEED_BOOST_POTION_DURATION",
    "FIRE_RATE_BOOST_POTION_INCREASE",
    "FIRE_RATE_BOOST_POTION_DURATION",
)


# Instant effects
class InstantEffectType(Enum):
    """Stores the type of instant effects that can be applied to an entity."""

    HEALTH = "health"
    ARMOUR = "armour"


# Status effect types
class StatusEffectType(Enum):
    """Stores the type of status effects that can be applied to an entity."""

    HEALTH = "health"
    ARMOUR = "armour"
    SPEED = "speed"
    FIRE_RATE = "fire rate"
    # BURN = "burn"


@dataclass
class ConsumableData:
    """
    The base class for constructing a consumable with multiple levels.

    name: str
        The name of the consumable.
    texture: arcade.Texture
        The texture for this consumable.
    level_limit: int
        The maximum level this consumable can go to.
    instant: Sequence[InstantData]
        The instant effects that this consumable gives.
    status_effects: Sequence[StatusEffectData]
        The status effects that this consumable gives.
    """

    name: str = field(kw_only=True)
    texture: arcade.Texture = field(kw_only=True)
    level_limit: int = field(kw_only=True)
    instant: Sequence[InstantData] = field(kw_only=True, default_factory=list)
    status_effects: Sequence[StatusEffectData] = field(
        kw_only=True, default_factory=list
    )


@dataclass
class InstantData:
    """
    Stores the data for an individual instant effect applied by a consumable.

    instant_type: InstantEffect
        The type of instant effect that is applied.
    increase: Callable[[int], float]
        The exponential lambda function which calculates the next level's value based on
        the current level.
    """

    instant_type: InstantEffectType = field(kw_only=True)
    increase: Callable[[int], float] = field(kw_only=True)


@dataclass
class StatusEffectData:
    """
    Stores the data for an individual status effect applied by a consumable.

    status_type: StatusEffectType
        The type of status effect that is applied.
    increase: Callable[[int], float]
        The exponential lambda function which calculates the next level's value based on
        the current level.
    duration: Callable[[int], float]
        The exponential lambda function which calculates the next level's duration based
        on the current level.
    duration: float
        The duration of this status effect.
    """

    status_type: StatusEffectType = field(kw_only=True)
    increase: Callable[[int], float] = field(kw_only=True)
    duration: Callable[[int], float] = field(kw_only=True)


# Other consumable constants
HEALTH_POTION_INCREASE = 20
ARMOUR_POTION_INCREASE = 10
HEALTH_BOOST_POTION_INCREASE = 50
HEALTH_BOOST_POTION_DURATION = 10
ARMOUR_BOOST_POTION_INCREASE = 10
ARMOUR_BOOST_POTION_DURATION = 10
SPEED_BOOST_POTION_INCREASE = 200
SPEED_BOOST_POTION_DURATION = 5
FIRE_RATE_BOOST_POTION_INCREASE = -0.5
FIRE_RATE_BOOST_POTION_DURATION = 5
