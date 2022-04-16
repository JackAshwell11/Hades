from __future__ import annotations

from dataclasses import dataclass

# Builtin
from enum import Enum
from typing import TYPE_CHECKING

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


# Status effect types
class StatusEffectType(Enum):
    """Stores the type of status effects that can be applied to the player."""

    HEALTH = "health"
    ARMOUR = "armour"
    SPEED = "speed"
    FIRE_RATE = "fire rate"


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


@dataclass
class BaseData:
    """
    The base class for constructing an entity. Only fill out some keyword arguments
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

    entity_data: EntityData
    player_data: PlayerData | None = None
    enemy_data: EnemyData | None = None
    ranged_attack_data: AttackData | None = None
    melee_attack_data: AttackData | None = None
    area_of_effect_attack_data: AreaOfEffectAttackData | None = None

    def get_all_attacks(self) -> list[AttackData]:
        """Returns all the attacks the entity has."""
        return list(
            filter(
                None,
                [
                    self.ranged_attack_data,
                    self.melee_attack_data,
                    self.area_of_effect_attack_data,
                ],
            )
        )


@dataclass
class EntityData:
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


@dataclass
class PlayerData:
    """
    Stores data about a specific player type.

    melee_range: int
        The amount of tiles the player can attack within using a melee attack.
    melee_degree: int
        The degree that the player's melee attack is limited to.
    """

    melee_range: int
    melee_degree: int


@dataclass
class EnemyData:
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


@dataclass
class AttackData:
    """
    Stores general data about an entity's attack.

    attack_type: AttackAlgorithmType
        The attack algorithm type that this dataclass describes.
    damage: int
        The damage the entity deals.
    """

    attack_type: AttackAlgorithmType
    damage: int


@dataclass
class AreaOfEffectAttackData(AttackData):
    """
    Stores data about an entity's area of effect attack.

    damage: int
        The damage the entity deals.
    range: int
        The range an area of effect attack deals has. This is the radius of the circle,
        not the diameter.
    """

    range: int


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
    player_data=PlayerData(
        3,
        60,
    ),
    ranged_attack_data=AttackData(AttackAlgorithmType.RANGED, 10),
    melee_attack_data=AttackData(AttackAlgorithmType.MELEE, 10),
    area_of_effect_attack_data=AreaOfEffectAttackData(
        AttackAlgorithmType.AREA_OF_EFFECT, 10, 3
    ),
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
    enemy_data=EnemyData(
        5,
        3,
        AIMovementType.FOLLOW,
    ),
    ranged_attack_data=AttackData(AttackAlgorithmType.RANGED, 5),
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
MELEE_RESOLUTION = 10
HEALTH_BAR_OFFSET = 40
ARMOUR_BAR_OFFSET = 32
