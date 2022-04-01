from __future__ import annotations

# Builtin
from enum import Enum, IntEnum
from typing import NamedTuple

# Pip
import arcade

# Custom
from textures import moving_textures


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
    max_velocity: int
        The max speed that the player can go.
    attack_cooldown: int
        The time duration between attacks.
    damage: int
        The damage the player deals.
    armour_regen: bool
        Whether the player regenerates armour or not.
    armour_regen_cooldown: int
        The time between armour regenerations.
    melee_range: int
        The amount of tiles the player can attack within using a melee attack.
    melee_degree: int
        The degree that the player's melee attack is limited to.
    """

    health: int
    armour: int
    textures: dict[str, list[list[arcade.Texture]]]
    max_velocity: int
    attack_cooldown: int
    damage: int
    armour_regen: bool
    armour_regen_cooldown: int
    melee_range: int
    melee_degree: int


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
    max_velocity: int
        The max speed that the enemy can go.
    attack_cooldown: int
        The time duration between attacks.
    damage: int
        The damage the enemy deals.
    armour_regen: bool
        Whether the enemy regenerates armour or not.
    armour_regen_cooldown: int
        The time between armour regenerations.
    view_distance: int
        The amount of tiles the enemy can see too.
    attack_range: int
        The amount of tiles the enemy can attack within.
    """

    name: str
    health: int
    armour: int
    textures: dict[str, list[list[arcade.Texture]]]
    max_velocity: int
    attack_cooldown: int
    damage: int
    armour_regen: bool
    armour_regen_cooldown: int
    view_distance: int
    attack_range: int


# Player characters
PLAYER = PlayerType(100, 20, moving_textures["player"], 200, 1, 10, True, 1, 3, 60)

# Enemy characters
ENEMY1 = EnemyType("enemy1", 10, 10, moving_textures["enemy"], 50, 1, 5, True, 3, 5, 3)
ENEMY2 = EnemyType("enemy2", 10, 10, moving_textures["enemy"], 50, 1, 5, True, 3, 5, 3)

# Status effect constants
HEALTH_BOOST_POTION_INCREASE = 50
HEALTH_BOOST_POTION_DURATION = 10

# Other entity constants
FACING_RIGHT = 0
FACING_LEFT = 1
MOVEMENT_FORCE = 1000000
ARMOUR_REGEN_WAIT = 5
ARMOUR_REGEN_AMOUNT = 1
BULLET_VELOCITY = 300
BULLET_OFFSET = 30
