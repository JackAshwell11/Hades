from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Sequence

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


# Player upgrades types
class PlayerUpgradeType(Enum):
    """Stores the types of upgrades that can be applied to the player."""

    HEALTH = "health"


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
    player_data: PlayerData | None = field(kw_only=True, default=None)
    enemy_data: EnemyData | None = field(kw_only=True, default=None)
    ranged_attack_data: RangedAttackData | None = field(kw_only=True, default=None)
    melee_attack_data: MeleeAttackData | None = field(kw_only=True, default=None)
    area_of_effect_attack_data: AreaOfEffectAttackData | None = field(
        kw_only=True, default=None
    )

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
    The base class for storing general data about an entity. This stuff should not
    change between entity levels.

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
    armour_regen: bool
        Whether the entity regenerates armour or not.
    armour_regen_cooldown: int
        The time between armour regenerations.
    """

    name: str = field(kw_only=True)
    health: int = field(kw_only=True)
    armour: int = field(kw_only=True)
    textures: dict[str, list[list[arcade.Texture]]] = field(kw_only=True)
    max_velocity: int = field(kw_only=True)
    armour_regen: bool = field(kw_only=True)
    armour_regen_cooldown: int = field(kw_only=True)


@dataclass
class PlayerData:
    """
    Stores data about a specific player type.

    melee_range: int
        The amount of tiles the player can attack within using a melee attack.
    melee_degree: int
        The degree that the player's melee attack is limited to.
    upgrade_data: Sequence[UpgradeData]
        The upgrades that are available to the player.
    """

    melee_range: int = field(kw_only=True)
    melee_degree: int = field(kw_only=True)
    upgrade_data: Sequence[UpgradeData] = field(
        kw_only=True, default_factory=lambda: [].copy()
    )


@dataclass
class UpgradeData:
    """
    Stores an upgrade that is available to the player.

    level_type: PlayerUpgradeType
        The type of upgrade this instance represents.
    level_one: int | None
        The first upgrade available for this type.
    level_two: int | None
        The second upgrade available for this type.
    level_three: int | None
        The third upgrade available for this type.
    """

    level_type: PlayerUpgradeType
    level_one: int | None = field(kw_only=True, default=None)
    level_two: int | None = field(kw_only=True, default=None)
    level_three: int | None = field(kw_only=True, default=None)


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

    view_distance: int = field(kw_only=True)
    attack_range: int = field(kw_only=True)
    movement_algorithm: AIMovementType = field(kw_only=True)


@dataclass
class AttackData:
    """
    The base class for storing data about an entity's attack.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    """

    damage: int = field(kw_only=True)
    attack_cooldown: int = field(kw_only=True)
    attack_type: AttackAlgorithmType


@dataclass
class RangedAttackData(AttackData):
    """
    Stores data about an entity's ranged attack.

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
    """
    Stores data about an entity's melee attack.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    """

    attack_type: AttackAlgorithmType = AttackAlgorithmType.MELEE


@dataclass
class AreaOfEffectAttackData(AttackData):
    """
    Stores data about an entity's area of effect attack.

    damage: int
        The damage the entity deals.
    attack_cooldown: int
        The time duration between attacks.
    attack_range: int
        The range an area of effect attack deals has. This is the radius of the circle,
        not the diameter.
    """

    attack_range: int = field(kw_only=True)
    attack_type: AttackAlgorithmType = AttackAlgorithmType.AREA_OF_EFFECT


# Player characters
PLAYER = BaseData(
    EntityData(
        name="player",
        health=100,
        armour=20,
        textures=moving_textures["player"],
        max_velocity=200,
        armour_regen=True,
        armour_regen_cooldown=1,
    ),
    player_data=PlayerData(
        melee_range=3,
        melee_degree=60,
        upgrade_data=[
            UpgradeData(
                PlayerUpgradeType.HEALTH, level_one=5, level_two=10, level_three=15
            )
        ],
    ),
    ranged_attack_data=RangedAttackData(damage=10, attack_cooldown=1, max_range=10),
    melee_attack_data=MeleeAttackData(damage=10, attack_cooldown=1),
    area_of_effect_attack_data=AreaOfEffectAttackData(
        damage=10, attack_cooldown=1, attack_range=3
    ),
)

# Enemy characters
ENEMY1 = BaseData(
    EntityData(
        name="enemy1",
        health=10,
        armour=10,
        textures=moving_textures["enemy"],
        max_velocity=50,
        armour_regen=True,
        armour_regen_cooldown=3,
    ),
    enemy_data=EnemyData(
        view_distance=5, attack_range=3, movement_algorithm=AIMovementType.FOLLOW
    ),
    ranged_attack_data=RangedAttackData(damage=5, attack_cooldown=1, max_range=10),
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
