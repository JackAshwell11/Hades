from __future__ import annotations

# Builtin
from enum import IntEnum
from typing import NamedTuple

# Pip
import arcade

# Custom
from textures import moving_textures


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


# Entity types
class PlayerType(NamedTuple):
    """
    Stores data about a specific player type.

    health: int
        The player's health.
    armour: int
        The player's armour.
    textures: dict[str, list[list[arcade.Texture]]]
        The textures which represent this player.
    movement_force: int
        The force applied to the player when it moves.
    melee_range: int
        The amount of tiles the player can attack within using a melee attack.
    melee_degree: int
        The degree that the player's melee attack is limited to.
    damage: int
        The damage the entity deals.
    """

    health: int
    armour: int
    textures: dict[str, list[list[arcade.Texture]]]
    movement_force: int
    melee_range: int
    melee_degree: int
    damage: int


class EnemyType(NamedTuple):
    """
    Stores data about a specific enemy type.

    name: str
        The name of the enemy.
    health: int
        The enemy's health.
    armour: int
        The enemy's armour.
    textures: dict[str, list[list[arcade.Texture]]]
        The textures which represent this enemy.
    movement_force: int
        The force applied to the enemy when it moves.
    view_distance: int
        The amount of tiles the enemy can see too.
    attack_range: int
        The amount of tiles the enemy can attack within.
    damage: int
        The damage the enemy deals.
    """

    name: str
    health: int
    armour: int
    textures: dict[str, list[list[arcade.Texture]]]
    movement_force: int
    view_distance: int
    attack_range: int
    damage: int


PLAYER = PlayerType(100, 20, moving_textures["player"], 10000, 3, 60, 10)
ENEMY1 = EnemyType("enemy1", 10, 10, moving_textures["enemy"], 20, 5, 3, 5)
ENEMY2 = EnemyType("enemy2", 10, 10, moving_textures["enemy"], 20, 5, 3, 5)

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

# Inventory constants
INVENTORY_WIDTH = 5
INVENTORY_HEIGHT = 3

# Item constants
CONSUMABLES = [TileType.HEALTH_POTION]
HEALTH_POTION_INCREASE = 10
