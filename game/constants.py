from __future__ import annotations

# Builtin
from enum import IntEnum

# Pip
import arcade


# Tile types
class TileType(IntEnum):
    """Stores the ID of each tile in the game map."""

    EMPTY = 0
    FLOOR = 1
    WALL = 2
    PLAYER = 3
    ENEMY = 4
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

# Map generation constants
BASE_MAP_WIDTH = 30
BASE_MAP_HEIGHT = 20
BASE_SPLIT_COUNT = 5
BASE_ENEMY_COUNT = 7
MIN_CONTAINER_SIZE = 7
MIN_ROOM_SIZE = 6
HALLWAY_SIZE = 5

# Debug constants
DEBUG_LINES = False
DEBUG_GAME = True
DEBUG_VIEW_DISTANCE = arcade.color.RED
DEBUG_ATTACK_DISTANCE = arcade.color.BLUE

# Sprite sizes
SPRITE_SCALE = 2.5
SPRITE_SIZE = 16 * SPRITE_SCALE

# Physics constants
DAMPING = 0

# Entity constants
FACING_RIGHT = 0
FACING_LEFT = 1

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
PLAYER_ATTACK_RANGE = 3
PLAYER_MELEE_DEGREE = 60
ENEMY_ATTACK_RANGE = 3
PLAYER_DAMAGE = 10
ENEMY_DAMAGE = 5
