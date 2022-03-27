from __future__ import annotations

# Builtin
from typing import NamedTuple

# Pip
import arcade

# Custom
from textures import moving_textures


# Base player type
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


# Base enemy type
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


# Player characters
PLAYER = PlayerType(100, 20, moving_textures["player"], 10000, 3, 60, 10)

# Enemy characters
ENEMY1 = EnemyType("enemy1", 10, 10, moving_textures["enemy"], 20, 5, 3, 5)
ENEMY2 = EnemyType("enemy2", 10, 10, moving_textures["enemy"], 20, 5, 3, 5)

# Other entity constants
FACING_RIGHT = 0
FACING_LEFT = 1
ATTACK_COOLDOWN = 1
BULLET_VELOCITY = 300
BULLET_OFFSET = 30
