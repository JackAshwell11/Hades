"""Stores various constants related to game objects and dataclasses used for
constructing the game object."""
from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

# Pip
import arcade

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = (
    "ARMOUR_INDICATOR_BAR_COLOR",
    "ARMOUR_REGEN_AMOUNT",
    "ARMOUR_REGEN_WAIT",
    "AreaOfEffectAttackData",
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
    "MeleeAttackData",
    "ObjectID",
    "PlayerData",
    "PlayerSectionUpgradeData",
    "RangedAttackData",
    "SPRITE_SCALE",
    "SPRITE_SIZE",
    "StatusEffectData",
    "StatusEffectType",
)


# Object IDs
class ObjectID(Enum):
    """Stores the ID of each game object to make checking more efficient."""

    BASE = "base"
    PLAYER = "player"
    ENEMY = "enemy"
    TILE = "tile"


# Entity attribute types
class EntityAttributeType(Enum):
    """Stores the types of attributes which an entity can have."""

    HEALTH = "health"
    ARMOUR = "armour"
    SPEED = "speed"
    REGEN_COOLDOWN = "regen cooldown"
    FIRE_RATE_MULTIPLIER = "fire rate multiplier"
    MONEY = "money"
    # POTION_DURATION = "potion duration"
    # MELEE_ATTACK = "melee attack"
    # AREA_OF_EFFECT_ATTACK = "area of effect attack"
    # RANGED_ATTACK = "ranged attack"


# Entity attribute sections types
class EntityAttributeSectionType(Enum):
    """Stores the sections which group all the entity attributes into similar types."""

    ENDURANCE = [EntityAttributeType.HEALTH, EntityAttributeType.SPEED]
    DEFENCE = [EntityAttributeType.ARMOUR, EntityAttributeType.REGEN_COOLDOWN]
    STRENGTH = []  # noqa
    INTELLIGENCE = []  # noqa


# Attack algorithms
class AttackAlgorithmType(Enum):
    """Stores the different types of attack algorithms that exist."""

    BASE = "base"
    RANGED = "ranged"
    MELEE = "melee"
    AREA_OF_EFFECT = "area of effect"


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
    FIRE_RATE = EntityAttributeType.FIRE_RATE_MULTIPLIER
    # BURN = "burn"


@dataclass
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

    entity_data: EntityData = field(kw_only=True)
    player_data: PlayerData | None = field(kw_only=True, default=None)
    enemy_data: EnemyData | None = field(kw_only=True, default=None)
    attacks: dict[AttackAlgorithmType, AttackData] = field(
        kw_only=True, default_factory=dict
    )


@dataclass
class EntityData:
    """The base class for storing general data about an entity. This stuff should not
    change between entity levels.

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

    name: str = field(kw_only=True)
    textures: dict[str, list[list[arcade.Texture]]] = field(kw_only=True)
    armour_regen: bool = field(kw_only=True)
    level_limit: int = field(kw_only=True)
    attribute_data: dict[EntityAttributeType, EntityAttributeData] = field(kw_only=True)


@dataclass
class EntityAttributeData:
    """Stores an attribute that is available to the entity.

    increase: Callable[[int], float]
        The exponential lambda function which calculates the next level's value based on
        the current level.
    maximum: bool
        Whether this attribute has a maximum value or not.
    upgradable: bool
        Whether this attribute is upgradable or not.
    status_effect: bool
        Whether this attribute can have a status effect applied to it or not.
    variable: bool
        Whether this attribute can change from it current value or not.
    """

    increase: Callable[[int], float] = field(kw_only=True)
    maximum: bool = field(kw_only=True, default=True)
    upgradable: bool = field(kw_only=True, default=False)
    status_effect: bool = field(kw_only=True, default=False)
    variable: bool = field(kw_only=True, default=False)


@dataclass
class PlayerData:
    """Stores data about a specific player type.

    melee_degree: int
        The degree that the player's melee attack is limited to.
    section_upgrade_data: list[PlayerSectionUpgradeData]
        The section upgrades that are available to the player.
    """

    melee_degree: int = field(kw_only=True)
    section_upgrade_data: list[PlayerSectionUpgradeData] = field(kw_only=True)


@dataclass
class PlayerSectionUpgradeData:
    """Stores a section upgrade that is available to the player. If the cost function is
    set to -1, then the section cannot be upgraded.

    section_type: EntityAttributeSectionType
        The type of entity attribute section this data represents.
    cost: Callable[[int], float]
        The exponential lambda function which calculates the next level's cost based on
        the current level.
    """

    section_type: EntityAttributeSectionType = field(kw_only=True)
    cost: Callable[[int], float] = field(kw_only=True)


@dataclass
class EnemyData:
    """Stores data about a specific enemy type.

    view_distance: int
        The amount of tiles the enemy can see too.
    """

    view_distance: int = field(kw_only=True)


@dataclass
class AttackData:
    """The base class for storing data about an entity's attack. This should not be
    initialised on its own.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    attack_range: int
        The range this attack has. This is the radius of the circle, not the diameter.
    """

    damage: int = field(kw_only=True)
    attack_cooldown: int = field(kw_only=True)
    attack_range: int = field(kw_only=True)


@dataclass
class RangedAttackData(AttackData):
    """Stores data about an entity's ranged attack.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    max_range: int
        The max range of the bullet.
    """

    max_range: int = field(kw_only=True)


@dataclass
class MeleeAttackData(AttackData):
    """Stores data about an entity's melee attack.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    """


@dataclass
class AreaOfEffectAttackData(AttackData):
    """Stores data about an entity's area of effect attack.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    """


@dataclass
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

    name: str = field(kw_only=True)
    texture: arcade.Texture = field(kw_only=True)
    level_limit: int = field(kw_only=True)
    instant: Sequence[InstantData] = field(kw_only=True, default_factory=list)
    status_effects: Sequence[StatusEffectData] = field(
        kw_only=True, default_factory=list
    )


@dataclass
class InstantData:
    """Stores the data for an individual instant effect applied by a consumable.

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

    status_type: StatusEffectType = field(kw_only=True)
    increase: Callable[[int], float] = field(kw_only=True)
    duration: Callable[[int], float] = field(kw_only=True)


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
