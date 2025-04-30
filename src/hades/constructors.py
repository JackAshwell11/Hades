"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Pip
from arcade import color, load_texture

# Custom
from hades_extensions import load_hitbox
from hades_extensions.ecs import GameObjectType
from hades_extensions.ecs.components import Armour, Health

if TYPE_CHECKING:
    from arcade import Texture
    from arcade.types.color import RGBA255

    from hades_extensions.ecs import ComponentBase

__all__ = ("GameObjectConstructor", "game_object_constructors", "texture_cache")


# The cache for all Arcade textures
texture_cache: dict[str, Texture] = {}

# Get the logger
logger = logging.getLogger(__name__)


@dataclass()
class GameObjectConstructor:
    """Represents a constructor for a game object.

    Args:
        name: The game object's name.
        description: The game object's description.
        game_object_type: The game object's type.
        depth: The game object's OpenGL rendering depth.
        texture_paths: The paths to the game object's textures.
        progress_bars: The game object's progress bars.
    """

    name: str
    description: str
    game_object_type: GameObjectType
    depth: int
    texture_paths: list[str]
    progress_bars: dict[type[ComponentBase], tuple[int, float, RGBA255]] = field(
        default_factory=dict,
    )

    def __post_init__(self: GameObjectConstructor) -> None:
        """Post-initialise the object."""
        logger.debug("Initialising game object constructor %s", self.name)
        for texture_path in self.texture_paths:
            if texture_path not in texture_cache:
                logger.debug("Loading texture %s", texture_path)
                texture_cache[texture_path] = load_texture(texture_path)
        logger.debug("Loading hitbox for %s", self.name)
        load_hitbox(
            self.game_object_type,
            texture_cache[self.texture_paths[0]].hit_box_points,
        )
        logger.debug("Initialised game object constructor %s", self.name)


def wall_factory() -> GameObjectConstructor:
    """Create a wall game object constructor.

    Returns:
        A wall game object constructor.
    """
    return GameObjectConstructor(
        "Wall",
        "A wall that blocks movement.",
        GameObjectType.Wall,
        0,
        [":resources:wall.png"],
    )


def floor_factory() -> GameObjectConstructor:
    """Create a floor game object constructor.

    Returns:
        A floor game object constructor.
    """
    return GameObjectConstructor(
        "Floor",
        "A floor that allows movement.",
        GameObjectType.Floor,
        0,
        [":resources:floor.png"],
    )


def player_factory() -> GameObjectConstructor:
    """Create a player game object constructor.

    Returns:
        A player game object constructor.
    """
    return GameObjectConstructor(
        "Player",
        "The player character.",
        GameObjectType.Player,
        3,
        [":resources:player_idle.png"],
        {
            Armour: (0, 2, color.SILVER),
            Health: (1, 2, color.RED),
        },
    )


def enemy_factory() -> GameObjectConstructor:
    """Create an enemy game object constructor.

    Returns:
        An enemy game object constructor.
    """
    return GameObjectConstructor(
        "Enemy",
        "An enemy character.",
        GameObjectType.Enemy,
        3,
        [":resources:enemy_idle.png"],
        {
            Armour: (0, 1, color.SILVER),
            Health: (1, 1, color.RED),
        },
    )


def goal_factory() -> GameObjectConstructor:
    """Create a goal game object constructor.

    Returns:
        A goal game object constructor.
    """
    return GameObjectConstructor(
        "Goal",
        "The goal of the level.",
        GameObjectType.Goal,
        1,
        [":resources:armour_potion.png"],
    )


def health_potion_factory() -> GameObjectConstructor:
    """Create a health potion game object constructor.

    Returns:
        A health potion game object constructor.
    """
    return GameObjectConstructor(
        "Health Potion",
        "A potion that restores health.",
        GameObjectType.HealthPotion,
        1,
        [":resources:health_potion.png"],
    )


def chest_factory() -> GameObjectConstructor:
    """Create a chest game object constructor.

    Returns:
        A chest game object constructor.
    """
    return GameObjectConstructor(
        "Chest",
        "A chest that contains loot.",
        GameObjectType.Chest,
        1,
        [":resources:shop.png"],
    )


def bullet_factory() -> GameObjectConstructor:
    """Create a bullet game object constructor.

    Returns:
        A bullet game object constructor
    """
    return GameObjectConstructor(
        "Bullet",
        "A bullet that damages other game objects.",
        GameObjectType.Bullet,
        2,
        [":resources:bullet.png"],
    )


game_object_constructors: dict[GameObjectType, GameObjectConstructor] = {
    GameObjectType.Bullet: bullet_factory(),
    GameObjectType.Enemy: enemy_factory(),
    GameObjectType.Floor: floor_factory(),
    GameObjectType.Goal: goal_factory(),
    GameObjectType.Player: player_factory(),
    GameObjectType.Wall: wall_factory(),
    GameObjectType.HealthPotion: health_potion_factory(),
    GameObjectType.Chest: chest_factory(),
}

# TODO: Attack should be modified to accept multiple categories (e.g. ranged,
#  close, special), so `Default` will change
