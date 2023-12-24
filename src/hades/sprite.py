"""Manages the operations related to the sprite object."""

from __future__ import annotations

# Builtin
from pathlib import Path
from typing import TYPE_CHECKING

# Pip
from arcade import Sprite, Texture, load_texture, load_texture_pair

# Custom
from hades_extensions.game_objects import SPRITE_SCALE, SPRITE_SIZE

if TYPE_CHECKING:
    from hades.constructors import GameObjectType

__all__ = ("AnimatedSprite", "BiggerThanError", "HadesSprite", "grid_pos_to_pixel")

# Create the texture path
texture_path = Path(__file__).resolve().parent / "resources" / "textures"


class BiggerThanError(Exception):
    """Raised when a value is less than a required value."""

    def __init__(self: BiggerThanError) -> None:
        """Initialise the object."""
        super().__init__("The input must be bigger than or equal to 0.")


def grid_pos_to_pixel(x: int, y: int) -> tuple[float, float]:
    """Calculate the screen position based on a grid position.

    Args:
        x: The x position in the grid.
        y: The x position in the grid.

    Returns:
        The screen position of the grid position.

    Raises:
        BiggerThanError: If one or both of the inputs are negative.
    """
    # Check if the inputs are negative
    if x < 0 or y < 0:
        raise BiggerThanError

    # Calculate the position on screen
    return (
        x * SPRITE_SIZE + SPRITE_SIZE / 2,
        y * SPRITE_SIZE + SPRITE_SIZE / 2,
    )


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
        position: tuple[int, int],
        sprite_textures: list[str],
    ) -> None:
        """Initialise the object.

        Args:
            game_object: The game object's ID and type.
            position: The position of the sprite object in the grid.
            sprite_textures: The sprites' texture paths.
        """
        super().__init__(
            load_texture(texture_path.joinpath(sprite_textures[0])),
            SPRITE_SCALE,
            *grid_pos_to_pixel(*position),
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
        position: tuple[int, int],
        textures: list[str],
    ) -> None:
        """Initialise the object.

        Args:
            game_object: The game object's ID and type.
            position: The position of the sprite object in the grid.
            textures: The sprite's texture paths.
        """
        super().__init__(game_object, position, textures)
        self.sprite_textures: list[tuple[Texture, Texture]] = [
            load_texture_pair(texture_path.joinpath(texture)) for texture in textures
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
