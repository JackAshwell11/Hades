from __future__ import annotations

# Builtin
from enum import IntEnum


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
    SHOP = 6
    DEBUG_WALL = 9


# Room probabilities
BASE_ROOM = {
    "SMALL": 0.5,
    "LARGE": 0.5,
}
SMALL_ROOM = {
    "SMALL": 0.3,
    "LARGE": 0.7,
}
LARGE_ROOM = {
    "SMALL": 0.7,
    "LARGE": 0.3,
}

# Map generation distributions
ENEMY_DISTRIBUTION = {
    TileType.ENEMY: 1,
}
ITEM_DISTRIBUTION = {
    TileType.HEALTH_POTION: 0.8,
    TileType.SHOP: 0.2,
}

# Other map generation constants
BASE_MAP_WIDTH = 30
BASE_MAP_HEIGHT = 20
BASE_SPLIT_COUNT = 5
BASE_ENEMY_COUNT = 7
BASE_ITEM_COUNT = 3
MIN_CONTAINER_SIZE = 7
MIN_ROOM_SIZE = 6
HALLWAY_SIZE = 5
SAFE_SPAWN_RADIUS = 5
PLACE_TRIES = 5
