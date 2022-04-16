from __future__ import annotations

# Builtin
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

# Custom
from game.constants.generation import TileType
from game.entities.attack import AreaOfEffectAttack, MeleeAttack, RangedAttack
from game.entities.movement import FollowLineOfSight, Jitter, MoveAwayLineOfSight
from game.textures import moving_textures

if TYPE_CHECKING:
    import arcade


# Entity IDs
class EntityID(Enum):
    """Stores the ID of each enemy to make collision checking more efficient."""

    ENTITY = "entity"
    PLAYER = "player"
    ENEMY = "enemy"


# Movement algorithms
class AIMovementType(Enum):
    """Stores the different type of enemy movement algorithms that exist."""

    FOLLOW = FollowLineOfSight
    JITTER = Jitter
    MOVE_AWAY = MoveAwayLineOfSight


# Attack algorithms
class AttackAlgorithmType(Enum):
    """Stores the different types of attack algorithms that exist."""

    RANGED = RangedAttack
    MELEE = MeleeAttack
    AREA_OF_EFFECT = AreaOfEffectAttack


# Status effect types
class StatusEffectType(Enum):
    """Stores the type of status effects that can be applied to the player."""

    HEALTH = "health"
    ARMOUR = "armour"
    SPEED = "speed"
    FIRE_RATE = "fire rate"


# Base entity data
class EntityData(NamedTuple):
    """
    Stores general data about an entity.

    name: str
        The name of the entity.
    health: int
        The entity's health.
    armour: int
        The entity's armour.
    textures: dict[str, list[list[arcade.Texture]]]
        The textures which represent this entity.
    max_velocity: int
        The max speed that the entity can go.
    attack_cooldown: int
        The time duration between attacks.
    armour_regen: bool
        Whether the entity regenerates armour or not.
    armour_regen_cooldown: int
        The time between armour regenerations.
    """

    name: str
    health: int
    armour: int
    textures: dict[str, list[list[arcade.Texture]]]
    max_velocity: int
    attack_cooldown: int
    armour_regen: bool
    armour_regen_cooldown: int


# Base player data
class PlayerData(NamedTuple):
    """
    Stores data about a specific player type.

    melee_range: int
        The amount of tiles the player can attack within using a melee attack.
    melee_degree: int
        The degree that the player's melee attack is limited to.
    """

    melee_range: int
    melee_degree: int


# Base enemy data
class EnemyData(NamedTuple):
    """
    Stores data about a specific enemy type.

    view_distance: int
        The amount of tiles the enemy can see too.
    attack_range: int
        The amount of tiles the enemy can attack within.
    movement_algorithm: AIMovementType
        The movement algorithm that this enemy has.
    """

    view_distance: int
    attack_range: int
    movement_algorithm: AIMovementType


# Base ranged attack data
class RangedAttackData(NamedTuple):
    """
    Stores data about an entity's ranged attack.

    attack_type: AttackAlgorithmType
        The attack algorithm type.
    damage: int
        The damage the entity deals.
    """

    damage: int


# Base melee attack data
class MeleeAttackData(NamedTuple):
    """
    Stores data about an entity's ranged attack.

    attack_type: AttackAlgorithmType
        The attack algorithm type.
    damage: int
        The damage the entity deals.
    """

    damage: int


# Base area of effect attack data
class AreaOfEffectAttackData(NamedTuple):
    """
    Stores data about an entity's area of effect attack.

    attack_type: AttackAlgorithmType
        The attack algorithm type.
    damage: int
        The damage the entity deals.
    area_of_effect_range: int
        The range an area of effect attack deals has. This is the radius of the circle,
        not the diameter.
    """

    damage: int
    area_of_effect_range: int


# Base entity constructor
class BaseData(NamedTuple):
    """
    The base class for constructing an entity.

    entity_type: EntityType
        The general data about this entity.
    custom_data: PlayerType | EnemyType
        The specific data related to this entity
    attack_data: dict[AttackAlgorithmType, AttackData]
        The data about the entity's attacks.
    """

    entity_data: EntityData
    custom_data: PlayerData | EnemyData
    attack_data: dict[
        AttackAlgorithmType, RangedAttackData | MeleeAttackData | AreaOfEffectAttackData
    ]


# Player characters
PLAYER = BaseData(
    EntityData(
        "player",
        100,
        20,
        moving_textures["player"],
        200,
        1,
        True,
        1,
    ),
    PlayerData(3, 60),
    {
        AttackAlgorithmType.RANGED: RangedAttackData(10),
        AttackAlgorithmType.MELEE: MeleeAttackData(10),
        AttackAlgorithmType.AREA_OF_EFFECT: AreaOfEffectAttackData(10, 3),
    },
)

# Enemy characters
ENEMY1 = BaseData(
    EntityData(
        "enemy1",
        10,
        10,
        moving_textures["enemy"],
        50,
        1,
        True,
        3,
    ),
    EnemyData(5, 3, AIMovementType.FOLLOW),
    {AttackAlgorithmType.RANGED: RangedAttackData(5)},
)

# Sprite sizes
SPRITE_SCALE = 2.5
SPRITE_SIZE = 16 * SPRITE_SCALE

# Other entity constants
ENEMIES = [TileType.ENEMY]
MOVEMENT_FORCE = 1000000
FACING_RIGHT = 0
FACING_LEFT = 1
ARMOUR_REGEN_WAIT = 5
ARMOUR_REGEN_AMOUNT = 1
BULLET_VELOCITY = 300
BULLET_OFFSET = 30
MELEE_RESOLUTION = 10
HEALTH_BAR_OFFSET = 40
ARMOUR_BAR_OFFSET = 32
