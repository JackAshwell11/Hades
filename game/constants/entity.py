"""Stores various constants related to entities and the dataclasses used for
constructing the entities."""
from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from game.entities.attack import AreaOfEffectAttack, MeleeAttack, RangedAttack

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = (
    "ARMOUR_INDICATOR_BAR_COLOR",
    "ARMOUR_REGEN_AMOUNT",
    "ARMOUR_REGEN_WAIT",
    "AreaOfEffectAttackData",
    "AttackAlgorithmType",
    "AttackData",
    "BULLET_VELOCITY",
    "BaseData",
    "ENEMY_INDICATOR_BAR_OFFSET",
    "EnemyData",
    "EntityAttributeData",
    "EntityAttributeSectionType",
    "EntityAttributeType",
    "EntityData",
    "EntityID",
    "FACING_LEFT",
    "FACING_RIGHT",
    "HEALTH_INDICATOR_BAR_COLOR",
    "INDICATOR_BAR_BORDER_SIZE",
    "MELEE_RESOLUTION",
    "MOVEMENT_FORCE",
    "MeleeAttackData",
    "PlayerData",
    "PlayerSectionUpgradeData",
    "RangedAttackData",
    "SPRITE_SCALE",
    "SPRITE_SIZE",
)


# Entity IDs
class EntityID(Enum):
    """Stores the ID of each enemy to make collision checking more efficient."""

    ENTITY = "entity"
    PLAYER = "player"
    ENEMY = "enemy"


# Entity attribute types
class EntityAttributeType(Enum):
    """Stores the types of attributes which an entity can have."""

    HEALTH = "health"
    ARMOUR = "armour"
    SPEED = "speed"
    REGEN_COOLDOWN = "regen cooldown"
    # POTION_DURATION = "potion duration"
    # MELEE_ATTACK = "melee attack"
    # AREA_OF_EFFECT_ATTACK = "area of effect attack"
    # RANGED_ATTACK = "ranged attack"


# Entity attribute sections types
class EntityAttributeSectionType(Enum):
    """Stores the sections which group all the entity attributes into similar types."""

    ENDURANCE = [EntityAttributeType.HEALTH, EntityAttributeType.SPEED]
    DEFENCE = [EntityAttributeType.ARMOUR, EntityAttributeType.REGEN_COOLDOWN]
    STRENGTH = []
    INTELLIGENCE = []


# Attack algorithms
class AttackAlgorithmType(Enum):
    """Stores the different types of attack algorithms that exist."""

    RANGED = RangedAttack
    MELEE = MeleeAttack
    AREA_OF_EFFECT = AreaOfEffectAttack


@dataclass
class BaseData:
    """The base class for constructing an entity. Only fill out some keyword arguments
    since not all of them are needed.

    entity_type: EntityType
        The data specifying the entity's attributes.
    player_data: PlayerData | None
        The data about the player entity.
    enemy_data: EnemyData | None
        The data about the enemy entity.
    ranged_attack_data: AttackData | None
        The data about the entity's ranged attack.
    melee_attack_data: AttackData | None
        The data about the entity's melee attack.
    area_of_effect_attack_data: AreaOfEffectAttackData | None
        The data about the entity's area of effect attack.
    """

    entity_data: EntityData = field(kw_only=True)
    player_data: PlayerData | None = field(kw_only=True, default=None)
    enemy_data: EnemyData | None = field(kw_only=True, default=None)
    ranged_attack_data: RangedAttackData | None = field(kw_only=True, default=None)
    melee_attack_data: MeleeAttackData | None = field(kw_only=True, default=None)
    area_of_effect_attack_data: AreaOfEffectAttackData | None = field(
        kw_only=True, default=None
    )

    def get_all_attacks(self) -> Iterator[AttackData]:
        """Returns all the attacks the entity has.

        Returns
        -------
        Iterator[AttackData]
            An iterator containing all the valid attack types for this entity.
        """
        return (
            attack
            for attack in (
                self.ranged_attack_data,
                self.melee_attack_data,
                self.area_of_effect_attack_data,
            )
            if attack
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
    upgradable: bool
        Whether this attribute is upgradable or not.
    status_effect: bool
        Whether this attribute can have a status effect applied to it or not.
    variable: bool
        Whether this attribute can change from it current value or not.
    """

    increase: Callable[[int], float] = field(kw_only=True)
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
    attack_type: AttackAlgorithmType


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
    attack_type: AttackAlgorithmType = AttackAlgorithmType.RANGED


@dataclass
class MeleeAttackData(AttackData):
    """Stores data about an entity's melee attack.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    """

    attack_type: AttackAlgorithmType = AttackAlgorithmType.MELEE


@dataclass
class AreaOfEffectAttackData(AttackData):
    """Stores data about an entity's area of effect attack.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    """

    attack_type: AttackAlgorithmType = AttackAlgorithmType.AREA_OF_EFFECT


# Sprite sizes
SPRITE_SCALE = 0.4375
SPRITE_SIZE = 128 * SPRITE_SCALE

# Other entity constants
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
