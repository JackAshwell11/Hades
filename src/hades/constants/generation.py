"""Stores various constants related to the dungeon generation."""
from __future__ import annotations

# Custom
from enum import Enum, auto

# Builtin
from typing import NamedTuple

__all__ = (
    "HALLWAY_SIZE",
    "ITEM_DISTRIBUTION",
    "MIN_CONTAINER_SIZE",
    "MIN_ROOM_SIZE",
    "ITEM_PLACE_TRIES",
    "WALL_REPLACEABLE_TILES",
    "ROOM_RATIO",
    "TileType",
    "MAP_GENERATION_COUNTS",
    "GenerationConstantType",
    "MapGenerationConstant",
)


# Tile types  # TODO: CHANGING THIS TO ENUM SEEMS TO CREATE DIAGONALS THROUGH WALLS
class TileType(Enum):
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


# Map generation counts
class GenerationConstantType(Enum):
    """Stores the different types of map generation constants."""

    WIDTH = auto()
    HEIGHT = auto()
    SPLIT_ITERATION = auto()
    OBSTACLE_COUNT = auto()
    ITEM_COUNT = auto()


class MapGenerationConstant(NamedTuple):
    """Stores a map generation constant which can be calculated.

    base_value: int
        The base value for the exponential calculation.
    increase: float
        The percentage increase for the constant.
    max_value: int
        The max value for the exponential calculation.
    """

    base_value: int
    increase: float
    max_value: int


MAP_GENERATION_COUNTS = {
    GenerationConstantType.WIDTH: MapGenerationConstant(30, 1.2, 150),
    GenerationConstantType.HEIGHT: MapGenerationConstant(20, 1.2, 100),
    GenerationConstantType.SPLIT_ITERATION: MapGenerationConstant(5, 1.5, 25),
    GenerationConstantType.OBSTACLE_COUNT: MapGenerationConstant(50, 1.3, 200),
    GenerationConstantType.ITEM_COUNT: MapGenerationConstant(3, 1.1, 15),
}

# Map generation item distribution
ITEM_DISTRIBUTION = {
    TileType.HEALTH_POTION: 0.3,
    TileType.ARMOUR_POTION: 0.3,
    TileType.HEALTH_BOOST_POTION: 0.2,
    TileType.ARMOUR_BOOST_POTION: 0.1,
    TileType.SPEED_BOOST_POTION: 0.05,
    TileType.FIRE_RATE_BOOST_POTION: 0.05,
}

# Bsp split constants
MIN_CONTAINER_SIZE = 5
MIN_ROOM_SIZE = 4
ROOM_RATIO = 0.625

# Room, hallway and entity generation constants
WALL_REPLACEABLE_TILES = [TileType.EMPTY, TileType.OBSTACLE, TileType.DEBUG_WALL]
HALLWAY_SIZE = 5
ITEM_PLACE_TRIES = 5
