"""Manages the base classes used by all game objects."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from hades.constants.game_objects import SPRITE_SCALE
from hades.textures import grid_pos_to_pixel

if TYPE_CHECKING:
    from hades.game_objects.enums import GameObjectData

__all__ = ("GameObject",)


class GameObject(arcade.Sprite):
    """The base class for all game objects.

    Attributes
    ----------
    center_x: float
        The x position of the object on the screen.
    center_y: float
        The y position of the object on the screen.
    """

    def __init__(
        self: GameObject, x: int, y: int, game_object_data: GameObjectData
    ) -> None:
        """Initialise the object.

        Parameters
        ----------
        x: int
            The x position of the object in the game map.
        y: int
            The y position of the object in the game map.
        game_object_data: GameObjectData
            The data related to this game object.
        """
        super().__init__(scale=SPRITE_SCALE)
        self.center_x, self.center_y = grid_pos_to_pixel(x, y)
        self.game_object_data: GameObjectData = game_object_data
        print(game_object_data)

    def __repr__(self: GameObject) -> str:
        """Return a human-readable representation of this object."""
        return f"<{self.game_object_data.name} (Position={self.position})>"
