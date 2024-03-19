"""Manages the operations related to the sprite object."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import (
    Sprite,
    SpriteSolidColor,
    Texture,
    color,
)

# Custom
from hades_extensions.game_objects import (
    SPRITE_SCALE,
    SPRITE_SIZE,
    Vec2d,
    grid_pos_to_pixel,
)

if TYPE_CHECKING:
    from hades.constants import GameObjectType

__all__ = (
    "AnimatedSprite",
    "Bullet",
    "HadesSprite",
)


class Bullet(SpriteSolidColor):
    """Represents a bullet sprite object in the game."""

    def __init__(self: Bullet, position: Vec2d) -> None:
        """Initialise the object.

        Args:
            position: The position of the sprite object in the grid.
        """
        super().__init__(
            SPRITE_SIZE,
            SPRITE_SIZE,
            color=color.RED,
        )
        self.center_x, self.center_y = grid_pos_to_pixel(position)


class HadesSprite(Sprite):
    """Represents a sprite object in the game.

    Attributes:
        game_object_id: The game object's ID.
        game_object_type: The game object's type.
    """

    __slots__ = ("game_object_id", "game_object_type")

    def __init__(
        self: HadesSprite,
        game_object: tuple[int, GameObjectType],
        position: Vec2d,
        sprite_textures: list[Texture],
    ) -> None:
        """Initialise the object.

        Args:
            game_object: The game object's ID and type.
            position: The position of the sprite object in the grid.
            sprite_textures: The sprites' texture paths.
        """
        super().__init__(
            sprite_textures[0],
            SPRITE_SCALE,
            *grid_pos_to_pixel(position),
        )
        self.game_object_id, self.game_object_type = game_object

    def __repr__(self: HadesSprite) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<HadesSprite (Game object ID={self.game_object_id}) (Current"
            f" texture={self.texture})>"
        )


class AnimatedSprite(HadesSprite):
    """Represents an animated sprite object in the game.

    Attributes:
        sprite_textures: The sprite's textures.
    """

    __slots__ = ("sprite_textures",)

    def __init__(
        self: AnimatedSprite,
        game_object: tuple[int, GameObjectType],
        position: Vec2d,
        textures: list[Texture],
    ) -> None:
        """Initialise the object.

        Args:
            game_object: The game object's ID and type.
            position: The position of the sprite object in the grid.
            textures: The sprite's texture paths.
        """
        super().__init__(game_object, position, textures)
        self.sprite_textures: list[tuple[Texture, Texture]] = [
            (texture, texture.flip_left_right()) for texture in textures
        ]

    def __repr__(self: AnimatedSprite) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<AnimatedSprite (Game object ID={self.game_object_id}) (Current"
            f" texture={self.texture})>"
        )
