from __future__ import annotations

# Builtin
from enum import IntEnum

# Pip
import arcade


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

# Map generation constants
BASE_MAP_WIDTH = 30
BASE_MAP_HEIGHT = 20
BASE_SPLIT_COUNT = 5
BASE_ENEMY_COUNT = 7
BASE_ITEM_COUNT = 3
MIN_CONTAINER_SIZE = 7
MIN_ROOM_SIZE = 6
HALLWAY_SIZE = 5

# Map generation distributions
ENEMY_DISTRIBUTION = {
    TileType.ENEMY: 1,
}
ITEM_DISTRIBUTION = {
    TileType.HEALTH_POTION: 0.8,
    TileType.SHOP: 0.2,
}

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
ATTACK_COOLDOWN = 1
BULLET_VELOCITY = 300
BULLET_OFFSET = 30

# Player constants
PLAYER_HEALTH = 100
PLAYER_MOVEMENT_FORCE = 10000
PLAYER_MELEE_RANGE = 3
PLAYER_MELEE_DEGREE = 60
PLAYER_DAMAGE = 10
INVENTORY_WIDTH = 5
INVENTORY_HEIGHT = 3

# Enemy constants
ENEMY_HEALTH = 10
ENEMY_MOVEMENT_FORCE = 20
ENEMY_VIEW_DISTANCE = 5
ENEMY_ATTACK_RANGE = 3
ENEMY_DAMAGE = 5

# Item constants
CONSUMABLES = [TileType.HEALTH_POTION]
HEALTH_POTION_INCREASE = 10
