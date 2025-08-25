"""Stores all the constructors used to make the game objects."""

from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

# Pip
from arcade import color, load_texture

# Custom
from hades_extensions import load_hitbox
from hades_extensions.ecs import GameObjectType

if TYPE_CHECKING:
    from arcade import Texture
    from arcade.types.color import RGBA255

__all__ = (
    "GameObjectConstructor",
    "IconType",
    "game_object_constructors",
    "texture_cache",
)


# The cache for all Arcade textures
texture_cache: dict[str, Texture] = {}


class IconType(Enum):
    """Represents the different types of icons."""

    BULLET = "bullet.png"
    CHEST = "chest.png"
    ENEMY_IDLE = "enemy_idle.png"
    FLOOR = "floor.png"
    GOAL = "armour_potion.png"
    HEALTH = "health_potion.png"
    MELEE = "fire_rate_boost_potion.png"
    MONEY = "money.png"
    MULTI_BULLET = "multi_bullet.png"
    PLAYER_IDLE = "player_idle.png"
    POISON = "speed_boost_potion.png"
    REGENERATION = "regeneration.png"
    SHOP = "shop.png"
    SINGLE_BULLET = "health_boost_potion.png"
    SPECIAL = "armour_boost_potion.png"
    WALL = "wall.png"

    def get_texture(self) -> Texture:
        """Get the cached texture for this icon type.

        Returns:
            The cached texture for this icon type.
        """
        if (texture_path := f":resources:textures/{self.value}") not in texture_cache:
            texture_cache[texture_path] = load_texture(texture_path)
        return texture_cache[texture_path]


@dataclass()
class GameObjectConstructor:
    """Represents a constructor for a game object.

    Args:
        name: The game object's name.
        description: The game object's description.
        game_object_type: The game object's type.
        depth: The game object's OpenGL rendering depth.
        textures: The game object's textures.
        progress_bars: The game object's progress bars.
    """

    name: str
    description: str
    game_object_type: GameObjectType
    depth: int
    textures: list[IconType]
    progress_bars: list[tuple[tuple[float, float], RGBA255]] = field(
        default_factory=list,
    )

    def __post_init__(self: GameObjectConstructor) -> None:
        """Post-initialise the object."""
        load_hitbox(
            self.game_object_type,
            self.textures[0].get_texture().hit_box_points,
        )


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
        [IconType.WALL],
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
        [IconType.FLOOR],
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
        [IconType.PLAYER_IDLE, IconType.PLAYER_IDLE],
        # Note health bars should always be first
        [((4, 2), color.RED), ((4, 2), color.SILVER)],
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
        [IconType.ENEMY_IDLE, IconType.ENEMY_IDLE],
        # Note health bars should always be first
        [((1, 1), color.RED), ((1, 1), color.SILVER)],
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
        [IconType.GOAL],
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
        [IconType.HEALTH],
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
        [IconType.CHEST],
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
        [IconType.BULLET],
    )


def shop_factory() -> GameObjectConstructor:
    """Create a shop game object constructor.

    Returns:
        A shop game object constructor.
    """
    return GameObjectConstructor(
        "Shop",
        "A shop that sells items.",
        GameObjectType.Shop,
        1,
        [IconType.CHEST],
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
    GameObjectType.Shop: shop_factory(),
}
