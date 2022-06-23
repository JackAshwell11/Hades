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
from game.constants.generation import TileType
from game.entities.attack import AreaOfEffectAttack, MeleeAttack, RangedAttack

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = (
    "EntityID",
    "UpgradeAttribute",
    "UpgradeSection",
    "AttackAlgorithmType",
    "BaseData",
    "EntityData",
    "EntityUpgradeData",
    "AttributeUpgradeData",
    "PlayerData",
    "EnemyData",
    "AttackData",
    "RangedAttackData",
    "MeleeAttackData",
    "AreaOfEffectAttackData",
    "SPRITE_SCALE",
    "SPRITE_SIZE",
    "ENEMIES",
    "MOVEMENT_FORCE",
    "FACING_RIGHT",
    "FACING_LEFT",
    "ARMOUR_REGEN_WAIT",
    "ARMOUR_REGEN_AMOUNT",
    "BULLET_VELOCITY",
    "MELEE_RESOLUTION",
    "INDICATOR_BAR_BORDER_SIZE",
    "ENEMY_INDICATOR_BAR_OFFSET",
    "HEALTH_INDICATOR_BAR_COLOR",
    "ARMOUR_INDICATOR_BAR_COLOR",
)


# Entity IDs
class EntityID(Enum):
    """Stores the ID of each enemy to make collision checking more efficient."""

    ENTITY = "entity"
    PLAYER = "player"
    ENEMY = "enemy"


# Entity attribute upgrades
class UpgradeAttribute(Enum):
    """Stores the types of attributes for the entity which can be upgraded."""

    HEALTH = "health"
    ARMOUR = "armour"
    SPEED = "speed"
    REGEN_COOLDOWN = "regen cooldown"
    # POTION_DURATION = "potion duration"
    # MELEE_ATTACK = "melee attack"
    # AREA_OF_EFFECT_ATTACK = "area of effect attack"
    # RANGED_ATTACK = "ranged attack"


# Entity upgrade sections
class UpgradeSection(Enum):
    """Stores the sections that can be upgraded by the player improving various
    attributes."""

    ENDURANCE = "endurance"
    DEFENCE = "defence"
    STRENGTH = "strength"
    INTELLIGENCE = "intelligence"


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
    upgrade_level_limit: int
        The maximum level the entity's upgrades can be.
    upgrade_data: list[UpgradeData]
        The upgrades that are available to the entity.
    """

    name: str = field(kw_only=True)
    textures: dict[str, list[list[arcade.Texture]]] = field(kw_only=True)
    armour_regen: bool = field(kw_only=True)
    upgrade_level_limit: int = field(kw_only=True)
    upgrade_data: list[EntityUpgradeData] = field(kw_only=True)


@dataclass
class EntityUpgradeData:
    """Stores an upgrade that is available to the entity. If the cost function is set
    to.

    -1, then the upgrade does not exist for the entity.

    section_type: UpgradeSection
        The type of upgrade this instance represents.
    cost: Callable[[int], float]
        The exponential lambda function which calculates the next level's cost based on
        the current level.
    level_limit: int
        The maximum level this upgrade can go to.
    upgrades: list[AttributeUpgrade]
        The list of attribute upgrades which are included in this instance.
    """

    section_type: UpgradeSection = field(kw_only=True)
    cost: Callable[[int], float] = field(kw_only=True)
    upgrades: list[AttributeUpgradeData] = field(kw_only=True)


@dataclass
class AttributeUpgradeData:
    """Stores an attribute upgrade that is available to the entity.

    attribute_type: UpgradeAttribute
        The type of attribute which this upgrade targets.
    increase: Callable[[int], float]
        The exponential lambda function which calculates the next level's value based on
        the current level.
    """

    attribute_type: UpgradeAttribute = field(kw_only=True)
    increase: Callable[[int], float] = field(kw_only=True)


@dataclass
class PlayerData:
    """Stores data about a specific player type.

    melee_degree: int
        The degree that the player's melee attack is limited to.
    """

    melee_degree: int = field(kw_only=True)


@dataclass
class EnemyData:
    """Stores data about a specific enemy type.

    view_distance: int
        The amount of tiles the enemy can see too.
    """

    view_distance: int = field(kw_only=True)


@dataclass
class AttackData:
    """The base class for storing data about an entity's attack.

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
ENEMIES = [TileType.ENEMY]
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
