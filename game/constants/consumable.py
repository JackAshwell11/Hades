from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Sequence

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
    BURN = "burn"


@dataclass
class ConsumableData:
    """
    The base class for constructing a consumable with multiple levels.

    name: str
        The name of the consumable.
    levels: dict[int, ConsumableLevelData]
        A mapping of level number to consumable level data. Level 1 is mandatory, but
        other levels are optional.
    """

    name: str = field(kw_only=True)
    levels: dict[int, ConsumableLevelData] = field(kw_only=True)


@dataclass
class ConsumableLevelData:
    """
    Stores the data for an individual level for a consumable. A level can have both
    an instant increase and a status effect (or multiple status effects).

    texture: arcade.Texture
        The texture for this consumable level.
    instant: Sequence[InstantData]
        The instant effects that this level gives.
    status_effects: Sequence[StatusEffectData]
        The status effects that this level gives.
    """

    texture: arcade.Texture = field(kw_only=True)
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
    value: float
        The value that is applied with this instant effect.
    """

    instant_type: InstantEffectType = field(kw_only=True)
    value: float = field(kw_only=True)


@dataclass
class StatusEffectData:
    """
    Stores the data for an individual status effect applied by a consumable.

    status_type: StatusEffectType
        The type of status effect that is applied.
    value: float
        The value that is applied with this status effect.
    duration: float
        The duration of this status effect.
    """

    status_type: StatusEffectType = field(kw_only=True)
    value: float = field(kw_only=True)
    duration: float = field(kw_only=True)


# Base instant consumables
HEALTH_POTION = ConsumableData(
    name="health potion",
    levels={
        1: ConsumableLevelData(
            texture=non_moving_textures["items"][0],
            instant=[InstantData(instant_type=InstantEffectType.HEALTH, value=10)],
        ),
        2: ConsumableLevelData(
            texture=non_moving_textures["items"][0],
            instant=[InstantData(instant_type=InstantEffectType.HEALTH, value=20)],
        ),
        3: ConsumableLevelData(
            texture=non_moving_textures["items"][0],
            instant=[InstantData(instant_type=InstantEffectType.HEALTH, value=30)],
        ),
    },
)

ARMOUR_POTION = ConsumableData(
    name="armour potion",
    levels={
        1: ConsumableLevelData(
            texture=non_moving_textures["items"][1],
            instant=[InstantData(instant_type=InstantEffectType.ARMOUR, value=5)],
        ),
        2: ConsumableLevelData(
            texture=non_moving_textures["items"][1],
            instant=[InstantData(instant_type=InstantEffectType.ARMOUR, value=10)],
        ),
        3: ConsumableLevelData(
            texture=non_moving_textures["items"][1],
            instant=[InstantData(instant_type=InstantEffectType.ARMOUR, value=20)],
        ),
    },
)

# Base status effect consumables
HEALTH_BOOST_POTION = ConsumableData(
    name="health boost potion",
    levels={
        1: ConsumableLevelData(
            texture=non_moving_textures["items"][2],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.HEALTH,
                    value=25,
                    duration=5,
                ),
            ],
        ),
        2: ConsumableLevelData(
            texture=non_moving_textures["items"][2],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.HEALTH,
                    value=50,
                    duration=10,
                ),
            ],
        ),
        3: ConsumableLevelData(
            texture=non_moving_textures["items"][2],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.HEALTH,
                    value=75,
                    duration=15,
                ),
            ],
        ),
    },
)

ARMOUR_BOOST_POTION = ConsumableData(
    name="armour boost potion",
    levels={
        1: ConsumableLevelData(
            texture=non_moving_textures["items"][3],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.ARMOUR,
                    value=5,
                    duration=5,
                ),
            ],
        ),
        2: ConsumableLevelData(
            texture=non_moving_textures["items"][3],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.ARMOUR,
                    value=10,
                    duration=10,
                ),
            ],
        ),
        3: ConsumableLevelData(
            texture=non_moving_textures["items"][3],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.ARMOUR,
                    value=20,
                    duration=15,
                ),
            ],
        ),
    },
)

SPEED_BOOST_POTION = ConsumableData(
    name="speed boost potion",
    levels={
        1: ConsumableLevelData(
            texture=non_moving_textures["items"][4],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.SPEED,
                    value=25,
                    duration=2,
                ),
            ],
        ),
        2: ConsumableLevelData(
            texture=non_moving_textures["items"][4],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.SPEED,
                    value=50,
                    duration=5,
                ),
            ],
        ),
        3: ConsumableLevelData(
            texture=non_moving_textures["items"][4],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.SPEED,
                    value=100,
                    duration=10,
                ),
            ],
        ),
    },
)

FIRE_RATE_BOOST_POTION = ConsumableData(
    name="fire rate boost potion",
    levels={
        1: ConsumableLevelData(
            texture=non_moving_textures["items"][5],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.FIRE_RATE,
                    value=-0.5,
                    duration=2,
                ),
            ],
        ),
        2: ConsumableLevelData(
            texture=non_moving_textures["items"][5],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.FIRE_RATE,
                    value=-0.5,
                    duration=5,
                ),
            ],
        ),
        3: ConsumableLevelData(
            texture=non_moving_textures["items"][5],
            status_effects=[
                StatusEffectData(
                    status_type=StatusEffectType.FIRE_RATE,
                    value=-0.5,
                    duration=10,
                ),
            ],
        ),
    },
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
