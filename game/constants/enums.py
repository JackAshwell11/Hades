from __future__ import annotations

from enum import Enum, IntEnum


# Entity IDs
class EntityID(IntEnum):
    """Stores the ID of each enemy to make collision checking more efficient."""

    ENTITY = 0
    PLAYER = 1
    ENEMY = 2


# Status effect types
class StatusEffectType(Enum):
    """Stores the type of status effects that can be applied to the player."""

    HEALTH = "health"
    ARMOUR = "armour"
    SPEED = "speed"
    FIRE_RATE = "fire rate"


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
