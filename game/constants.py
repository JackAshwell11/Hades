from __future__ import annotations

# Builtin
from enum import IntEnum
from typing import NamedTuple


# Tile types
class TileType(IntEnum):
    """Stores the ID of each tile in the game map."""

    EMPTY = 0
    FLOOR = 1
    WALL = 2
    PLAYER = 3
    ENEMY = 4
    DEBUG_WALL = 9


# Room constants
class Room(NamedTuple):
    """
    Represents a template for a room in the game map.

    MIN_SIZE: int
        The minimum size a room can be.
    MAX_SIZE: int
        The maximum size a room can be.
    """

    MIN_SIZE: int = -1
    MAX_SIZE: int = -1


SMALL_ROOM = Room()
MEDIUM_ROOM = Room()
LARGER_ROOM = Room()

# ADD PROBABILITIES IN HERE AND NAMED TUPLE SIZES FOR HALLWAY

# Map generation constants
BASE_MAP_WIDTH = 30
BASE_MAP_HEIGHT = 20
BASE_SPLIT_COUNT = 2
BASE_ENEMY_COUNT = 5
MIN_CONTAINER_SIZE = 7
MIN_ROOM_SIZE = 5
HALLWAY_WIDTH = 5

# Debug constants
DEBUG_LINES = False
DEBUG_GAME = True

# Sprite sizes
SPRITE_SCALE = 2.5
SPRITE_WIDTH = 16 * SPRITE_SCALE
SPRITE_HEIGHT = 16 * SPRITE_SCALE

# Physics constants
DAMPING = 0

# Player constants
PLAYER_MOVEMENT_FORCE = 10000
PLAYER_HEALTH = 100

# Enemy constants
ENEMY_MOVEMENT_FORCE = 20
ENEMY_HEALTH = 10
ENEMY_VIEW_DISTANCE = 5

# Attack constants
ATTACK_COOLDOWN = 1
BULLET_VELOCITY = 300
BULLET_OFFSET = 30
ENEMY_ATTACK_RANGE = 3
PLAYER_DAMAGE = 10
ENEMY_DAMAGE = 5
