from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

# Custom
from game.constants.generation import TileType
from game.textures import non_moving_textures

if TYPE_CHECKING:
    import arcade


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
    level_one: ConsumableLevelData
        The data for the first level of the consumable.
    level_two: ConsumableLevelData | None
        The data for the second level of the consumable. This level is optional.
    level_three: ConsumableLevelData | None
        The data for the third level of the consumable. This level is optional.
    """

    name: str = field(kw_only=True)
    level_one: ConsumableLevelData = field(kw_only=True)
    level_two: ConsumableLevelData | None = field(kw_only=True, default=None)
    level_three: ConsumableLevelData | None = field(kw_only=True, default=None)


@dataclass
class ConsumableLevelData:
    """
    Stores the data for an individual level for a consumable. A level can have both
    an instant increase and a status effect (or multiple status effects).

    texture: arcade.Texture
        The texture for this consumable level.
    instant: int | None
        The instant effect that this level gives.
    status_effects: list[StatusEffectData] | None
        The status effects that this level gives.
    """

    texture: arcade.Texture = field(kw_only=True)
    instant: int | None = field(kw_only=True, default=None)
    status_effects: list[StatusEffectData] | None = field(kw_only=True, default=None)


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
    level_one=ConsumableLevelData(
        texture=non_moving_textures["items"][0],
        instant=10,
    ),
    level_two=ConsumableLevelData(
        texture=non_moving_textures["items"][0],
        instant=20,
    ),
    level_three=ConsumableLevelData(
        texture=non_moving_textures["items"][0],
        instant=30,
    ),
)

ARMOUR_POTION = ConsumableData(
    name="armour potion",
    level_one=ConsumableLevelData(
        texture=non_moving_textures["items"][1],
        instant=5,
    ),
    level_two=ConsumableLevelData(
        texture=non_moving_textures["items"][1],
        instant=10,
    ),
    level_three=ConsumableLevelData(
        texture=non_moving_textures["items"][1],
        instant=20,
    ),
)

# Base status effect consumables
HEALTH_BOOST_POTION = ConsumableData(
    name="health boost potion",
    level_one=ConsumableLevelData(
        texture=non_moving_textures["items"][2],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.HEALTH,
                value=25,
                duration=5,
            ),
        ],
    ),
    level_two=ConsumableLevelData(
        texture=non_moving_textures["items"][2],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.HEALTH,
                value=50,
                duration=10,
            ),
        ],
    ),
    level_three=ConsumableLevelData(
        texture=non_moving_textures["items"][2],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.HEALTH,
                value=75,
                duration=15,
            ),
        ],
    ),
)

ARMOUR_BOOST_POTION = ConsumableData(
    name="armour boost potion",
    level_one=ConsumableLevelData(
        texture=non_moving_textures["items"][3],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.ARMOUR,
                value=5,
                duration=5,
            ),
        ],
    ),
    level_two=ConsumableLevelData(
        texture=non_moving_textures["items"][3],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.ARMOUR,
                value=10,
                duration=10,
            ),
        ],
    ),
    level_three=ConsumableLevelData(
        texture=non_moving_textures["items"][3],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.ARMOUR,
                value=20,
                duration=15,
            ),
        ],
    ),
)

SPEED_BOOST_POTION = ConsumableData(
    name="speed boost potion",
    level_one=ConsumableLevelData(
        texture=non_moving_textures["items"][4],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.SPEED,
                value=25,
                duration=2,
            ),
        ],
    ),
    level_two=ConsumableLevelData(
        texture=non_moving_textures["items"][4],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.SPEED,
                value=50,
                duration=5,
            ),
        ],
    ),
    level_three=ConsumableLevelData(
        texture=non_moving_textures["items"][4],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.SPEED,
                value=100,
                duration=10,
            ),
        ],
    ),
)

FIRE_RATE_BOOST_POTION = ConsumableData(
    name="fire rate boost potion",
    level_one=ConsumableLevelData(
        texture=non_moving_textures["items"][5],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.FIRE_RATE,
                value=-0.5,
                duration=2,
            ),
        ],
    ),
    level_two=ConsumableLevelData(
        texture=non_moving_textures["items"][5],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.FIRE_RATE,
                value=-0.5,
                duration=5,
            ),
        ],
    ),
    level_three=ConsumableLevelData(
        texture=non_moving_textures["items"][5],
        status_effects=[
            StatusEffectData(
                status_type=StatusEffectType.FIRE_RATE,
                value=-0.5,
                duration=10,
            ),
        ],
    ),
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
