"""Stores various constants related to the dungeon generation."""
from __future__ import annotations

# Custom
from enum import IntEnum, auto

__all__ = (
    "BASE_ITEM_COUNT",
    "BASE_MAP_HEIGHT",
    "BASE_MAP_WIDTH",
    "BASE_OBSTACLE_COUNT",
    "BASE_SPLIT_ITERATION",
    "HALLWAY_SIZE",
    "ITEM_DISTRIBUTION",
    "MAX_ITEM_COUNT",
    "MAX_MAP_HEIGHT",
    "MAX_MAP_WIDTH",
    "MAX_OBSTACLE_COUNT",
    "MAX_SPLIT_ITERATION",
    "MIN_CONTAINER_SIZE",
    "MIN_ROOM_SIZE",
    "ITEM_PLACE_TRIES",
    "WALL_REPLACEABLE_TILES",
    "ROOM_RATIO",
    "TileType",
)


# Tile types
# noinspection PyArgumentList
# TODO REMOVE ABOVE LINE ONCE BUG FIXED
class TileType(IntEnum):
    """Stores the different types of tiles in the game map."""

    EMPTY = auto()
    FLOOR = auto()
    WALL = auto()
    OBSTACLE = auto()
    PLAYER = auto()
    HEALTH_POTION = auto()
    ARMOUR_POTION = auto()
    HEALTH_BOOST_POTION = auto()
    ARMOUR_BOOST_POTION = auto()
    SPEED_BOOST_POTION = auto()
    FIRE_RATE_BOOST_POTION = auto()
    DEBUG_WALL = auto()


# Map generation item distribution
ITEM_DISTRIBUTION = {
    TileType.HEALTH_POTION: 0.3,
    TileType.ARMOUR_POTION: 0.3,
    TileType.HEALTH_BOOST_POTION: 0.2,
    TileType.ARMOUR_BOOST_POTION: 0.1,
    TileType.SPEED_BOOST_POTION: 0.05,
    TileType.FIRE_RATE_BOOST_POTION: 0.05,
}

# Map generation counts
BASE_MAP_WIDTH = 30
BASE_MAP_HEIGHT = 20
BASE_SPLIT_ITERATION = 5
BASE_OBSTACLE_COUNT = 50
BASE_ITEM_COUNT = 3
MAX_MAP_WIDTH = 150
MAX_MAP_HEIGHT = 100
MAX_SPLIT_ITERATION = 25
MAX_OBSTACLE_COUNT = 200
MAX_ITEM_COUNT = 15

# Bsp split constants
MIN_CONTAINER_SIZE = 5
MIN_ROOM_SIZE = 4
ROOM_RATIO = 0.625

# Room, hallway and entity generation constants
WALL_REPLACEABLE_TILES = [TileType.EMPTY, TileType.OBSTACLE, TileType.DEBUG_WALL]
HALLWAY_SIZE = 5
ITEM_PLACE_TRIES = 5
