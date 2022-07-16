"""Stores various constants related to game objects and their dataclasses."""
from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

# Pip
import arcade

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = (
    "ARMOUR_INDICATOR_BAR_COLOR",
    "ARMOUR_REGEN_AMOUNT",
    "ARMOUR_REGEN_WAIT",
    "AttackAlgorithmType",
    "AttackData",
    "BULLET_VELOCITY",
    "BaseData",
    "ConsumableData",
    "ENEMY_INDICATOR_BAR_OFFSET",
    "EnemyData",
    "EntityAttributeData",
    "EntityAttributeSectionType",
    "EntityAttributeType",
    "EntityData",
    "FACING_LEFT",
    "FACING_RIGHT",
    "HEALTH_INDICATOR_BAR_COLOR",
    "INDICATOR_BAR_BORDER_SIZE",
    "InstantData",
    "InstantEffectType",
    "MELEE_RESOLUTION",
    "MOVEMENT_FORCE",
    "ObjectID",
    "PlayerData",
    "RangedAttackData",
    "SPRITE_SCALE",
    "SPRITE_SIZE",
    "StatusEffectData",
    "StatusEffectType",
)


# Object IDs
# noinspection PyArgumentList
# TODO REMOVE ABOVE LINE ONCE BUG FIXED
class ObjectID(Enum):
    """Stores the different types of game objects to make checking more efficient."""

    BASE = auto()
    PLAYER = auto()
    ENEMY = auto()
    TILE = auto()


# Entity attribute types
# noinspection PyArgumentList
# TODO REMOVE ABOVE LINE ONCE BUG FIXED
class EntityAttributeType(Enum):
    """Stores the types of attributes an entity can have."""

    HEALTH = auto()
    ARMOUR = auto()
    SPEED = auto()
    REGEN_COOLDOWN = auto()
    FIRE_RATE_PENALTY = auto()
    MONEY = auto()
    # POTION_DURATION = auto()
    # MELEE_ATTACK = auto()
    # AREA_OF_EFFECT_ATTACK = auto()
    # RANGED_ATTACK = auto()


# Entity attribute sections types
class EntityAttributeSectionType(Enum):
    """Stores the sections which group all the entity attributes into similar types."""

    ENDURANCE = [EntityAttributeType.HEALTH, EntityAttributeType.SPEED]
    DEFENCE = [EntityAttributeType.ARMOUR, EntityAttributeType.REGEN_COOLDOWN]
    # STRENGTH = []
    # INTELLIGENCE = []


# Attack algorithms
# noinspection PyArgumentList
# TODO REMOVE ABOVE LINE ONCE BUG FIXED
class AttackAlgorithmType(Enum):
    """Stores the different types of attack algorithms that exist."""

    BASE = auto()
    RANGED = auto()
    MELEE = auto()
    AREA_OF_EFFECT = auto()


# Instant effect types
class InstantEffectType(Enum):
    """Stores the type of instant effects that can be applied to an entity."""

    HEALTH = EntityAttributeType.HEALTH
    ARMOUR = EntityAttributeType.ARMOUR


# Status effect types
class StatusEffectType(Enum):
    """Stores the type of status effects that can be applied to an entity."""

    HEALTH = EntityAttributeType.HEALTH
    ARMOUR = EntityAttributeType.ARMOUR
    SPEED = EntityAttributeType.SPEED
    FIRE_RATE = EntityAttributeType.FIRE_RATE_PENALTY
    # BURN = "burn"
    # POISON = "poison"


@dataclass(frozen=True, kw_only=True, slots=True)
class BaseData:
    """The base class for constructing an entity.

    entity_type: EntityType
        The data specifying the entity's attributes.
    player_data: PlayerData | None
        The data about the player entity.
    enemy_data: EnemyData | None
        The data about the enemy entity.
    attacks: dict[AttackAlgorithmType, AttackData]
        The data about the entity's attacks.
    """

    entity_data: EntityData
    player_data: PlayerData | None = None
    enemy_data: EnemyData | None = None
    attacks: dict[AttackAlgorithmType, AttackData] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True, slots=True)
class EntityData:
    """The base class for storing general data about an entity.

    name: str
        The name of the entity.
    textures: dict[str, list[list[arcade.Texture]]]
        The textures which represent this entity.
    armour_regen: bool
        Whether the entity regenerates armour or not.
    level_limit: int
        The maximum level the entity can be.
    attribute_data: dict[EntityAttributeType, EntityAttributeData]
        The attributes that are available to this entity.
    """

    name: str
    textures: dict[str, list[list[arcade.Texture]]]
    armour_regen: bool
    level_limit: int
    attribute_data: dict[EntityAttributeType, EntityAttributeData]


@dataclass(frozen=True, kw_only=True, slots=True)
class EntityAttributeData:
    """Stores an attribute that is available to the entity.

    increase: Callable[[int], float]
        The exponential lambda function which calculates the next level's value based on
        the current level.
    maximum: bool
        Whether this attribute has a maximum value or not.
    status_effect: bool
        Whether this attribute can have a status effect applied to it or not.
    variable: bool
        Whether this attribute can change from it current value or not.
    """

    increase: Callable[[int], float]
    maximum: bool = True
    status_effect: bool = False
    variable: bool = False


@dataclass(frozen=True, kw_only=True, slots=True)
class PlayerData:
    """Stores data about a specific player type.

    melee_degree: int
        The degree that the player's melee attack is limited to.
    section_upgrade_data: dict[EntityAttributeSectionType, Callable[[int], float]]
        The section upgrades that are available to the player.
    """

    melee_degree: int
    section_upgrade_data: dict[EntityAttributeSectionType, Callable[[int], float]]


@dataclass(frozen=True, kw_only=True, slots=True)
class EnemyData:
    """Stores data about a specific enemy type.

    view_distance: int
        The amount of tiles the enemy can see too.
    """

    view_distance: int


@dataclass(frozen=True, kw_only=True, slots=True)
class AttackData:
    """Stores generalized data about an entity's attack.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    attack_range: int
        The range this attack has. This is the radius of the circle, not the diameter.
    extra: RangedAttackData | None
        The extra data about this entity's attack.
    """

    damage: int
    attack_cooldown: int
    attack_range: int
    extra: RangedAttackData | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class RangedAttackData:
    """Stores extra data about an entity's ranged attack.

    max_bullet_range: int
        The max range of the bullet.
    """

    max_bullet_range: int


# @dataclass(frozen=True, kw_only=True, slots=True)
# class MeleeAttackData:
#     """Stores data about an entity's melee attack."""
#
#
# @dataclass(frozen=True, kw_only=True, slots=True)
# class AreaOfEffectAttackData:
#     """Stores data about an entity's area of effect attack."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ConsumableData:
    """The base class for constructing a consumable with multiple levels.

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

    name: str
    texture: arcade.Texture
    level_limit: int
    instant: Sequence[InstantData] = field(default_factory=list)
    status_effects: Sequence[StatusEffectData] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True, slots=True)
class InstantData:
    """Stores the data for an individual instant effect applied by a consumable.

    instant_type: InstantEffect
        The type of instant effect that is applied.
    increase: Callable[[int], float]
        The exponential lambda function which calculates the next level's value based on
        the current level.
    """

    instant_type: InstantEffectType
    increase: Callable[[int], float]


@dataclass(frozen=True, kw_only=True, slots=True)
class StatusEffectData:
    """Stores the data for an individual status effect applied by a consumable.

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

    status_type: StatusEffectType
    increase: Callable[[int], float]
    duration: Callable[[int], float]


# Sprite sizes
SPRITE_SCALE = 0.4375
SPRITE_SIZE = 128 * SPRITE_SCALE

# Other game object constants
MOVEMENT_FORCE = 1000000
FACING_RIGHT = 0
FACING_LEFT = 1
ARMOUR_REGEN_WAIT = 5
ARMOUR_REGEN_AMOUNT = 1
BULLET_VELOCITY = 300
MELEE_RESOLUTION = 10
INDICATOR_BAR_BORDER_SIZE = 4
ENEMY_INDICATOR_BAR_OFFSET = 32
HEALTH_INDICATOR_BAR_COLOR = arcade.color.RED
ARMOUR_INDICATOR_BAR_COLOR = arcade.color.SILVER
