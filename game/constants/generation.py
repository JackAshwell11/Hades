"""Stores various constants related to the bsp generation."""
from __future__ import annotations

# Custom
from enum import IntEnum

__all__ = (
    "BASE_ENEMY_COUNT",
    "BASE_ITEM_COUNT",
    "BASE_MAP_HEIGHT",
    "BASE_MAP_WIDTH",
    "BASE_SPLIT_ITERATION",
    "ENEMY_DISTRIBUTION",
    "HALLWAY_SIZE",
    "ITEM_DISTRIBUTION",
    "MAX_ENEMY_COUNT",
    "MAX_ITEM_COUNT",
    "MAX_MAP_HEIGHT",
    "MAX_MAP_WIDTH",
    "MAX_SPLIT_ITERATION",
    "MIN_CONTAINER_SIZE",
    "MIN_ROOM_SIZE",
    "PLACE_TRIES",
    "SAFE_SPAWN_RADIUS",
    "TileType",
    "ROOM_RATIO",
)


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
    ARMOUR_POTION = 6
    HEALTH_BOOST_POTION = 7
    ARMOUR_BOOST_POTION = 8
    SPEED_BOOST_POTION = 9
    FIRE_RATE_BOOST_POTION = 10
    DEBUG_WALL = 11


# Map generation distributions
ENEMY_DISTRIBUTION = {
    TileType.ENEMY: 1,
}
ITEM_DISTRIBUTION = {
    TileType.HEALTH_POTION: 0.3,
    TileType.ARMOUR_POTION: 0.3,
    TileType.HEALTH_BOOST_POTION: 0.1,
    TileType.ARMOUR_BOOST_POTION: 0.1,
    TileType.SPEED_BOOST_POTION: 0.1,
    TileType.FIRE_RATE_BOOST_POTION: 0.1,
}

# Other map generation constants
BASE_MAP_WIDTH = 30
MAX_MAP_WIDTH = 150
BASE_MAP_HEIGHT = 20
MAX_MAP_HEIGHT = 100
BASE_SPLIT_ITERATION = 8
MAX_SPLIT_ITERATION = 25
BASE_ENEMY_COUNT = 7
MAX_ENEMY_COUNT = 35
BASE_ITEM_COUNT = 3
MAX_ITEM_COUNT = 15
MIN_CONTAINER_SIZE = 5
MIN_ROOM_SIZE = 4
ROOM_RATIO = 0.625
HALLWAY_SIZE = 5
SAFE_SPAWN_RADIUS = 5
PLACE_TRIES = 5
