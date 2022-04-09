from __future__ import annotations

# Builtin
from enum import Enum, IntEnum

# Custom
from entities.attack import AreaOfEffectAttack, MeleeAttack, RangedAttack
from entities.movement import FollowLineOfSight, Jitter, MoveAwayLineOfSight


# Tile types
class TileType(IntEnum):
    """Stores the ID of each tile in the game map."""

    NONE = -1
    EMPTY = 0
    FLOOR = 1
    WALL = 2
    PLAYER = 3
    ENEMY = 4
    HEALTH_POTION = 5
    HEALTH_BOOST_POTION = 6
    SHOP = 7
    DEBUG_WALL = 9


# Entity IDs
class EntityID(IntEnum):
    """Stores the ID of each enemy to make collision checking more efficient."""

    ENTITY = 0
    PLAYER = 1
    ENEMY = 2


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
