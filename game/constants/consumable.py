from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Sequence

# Custom
from game.constants.generation import TileType
from game.textures import non_moving_textures

if TYPE_CHECKING:
    import arcade


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
    # THORNS = "thorns"


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
    instant: Sequence[InstantData] = field(
        kw_only=True, default_factory=lambda: [].copy()
    )
    status_effects: Sequence[StatusEffectData] = field(
        kw_only=True, default_factory=lambda: [].copy()
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


# Base instant consumables
HEALTH_POTION = ConsumableData(
    name="health potion",
    texture=non_moving_textures["items"][0],
    level_limit=5,
    instant=[
        InstantData(
            instant_type=InstantEffectType.HEALTH,
            increase=lambda current_level: 10 * 1.5**current_level,
        ),
    ],
)

ARMOUR_POTION = ConsumableData(
    name="armour potion",
    texture=non_moving_textures["items"][1],
    level_limit=5,
    instant=[
        InstantData(
            instant_type=InstantEffectType.ARMOUR,
            increase=lambda current_level: 10 * 1.5**current_level,
        ),
    ],
)

# Base status effect consumables
HEALTH_BOOST_POTION = ConsumableData(
    name="health boost potion",
    texture=non_moving_textures["items"][2],
    level_limit=5,
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.HEALTH,
            increase=lambda current_level: 25 * 1.3**current_level,
            duration=lambda current_level: 5 * 1.3**current_level,
        )
    ],
)

ARMOUR_BOOST_POTION = ConsumableData(
    name="armour boost potion",
    texture=non_moving_textures["items"][3],
    level_limit=5,
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.ARMOUR,
            increase=lambda current_level: 10 * 1.3**current_level,
            duration=lambda current_level: 5 * 1.3**current_level,
        )
    ],
)

SPEED_BOOST_POTION = ConsumableData(
    name="speed boost potion",
    texture=non_moving_textures["items"][4],
    level_limit=5,
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.HEALTH,
            increase=lambda current_level: 25 * 1.3**current_level,
            duration=lambda current_level: 2 * 1.3**current_level,
        )
    ],
)

FIRE_RATE_BOOST_POTION = ConsumableData(
    name="fire rate boost potion",
    texture=non_moving_textures["items"][5],
    level_limit=5,
    status_effects=[
        StatusEffectData(
            status_type=StatusEffectType.HEALTH,
            increase=lambda current_level: -0.5,
            duration=lambda current_level: 2 * 1.3**current_level,
        )
    ],
)


# Other consumable constants
CONSUMABLES = [
    TileType.HEALTH_POTION,
    TileType.ARMOUR_POTION,
    TileType.HEALTH_BOOST_POTION,
    TileType.ARMOUR_BOOST_POTION,
    TileType.SPEED_BOOST_POTION,
    TileType.FIRE_RATE_BOOST_POTION,
]
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
