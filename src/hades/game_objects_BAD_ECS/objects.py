"""Manages the base classes used by all game objects."""
from __future__ import annotations

# Pip
import arcade

# Custom
from hades.constants_OLD.game_objects import SPRITE_SCALE
from hades.textures import grid_pos_to_pixel

# Builtin


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

    def __init__(self: GameObject, x: int, y: int, **kwargs) -> None:
        """Initialise the object.

        Parameters
        ----------
        x: int
            The x position of the object in the game map.
        y: int
            The y position of the object in the game map.
        """
        super().__init__(scale=SPRITE_SCALE, **kwargs)
        self.center_x, self.center_y = grid_pos_to_pixel(x, y)
        print("game object", kwargs)

    def __repr__(self: GameObject) -> str:
        """Return a human-readable representation of this object."""
        return f"<GameObject (Position={self.position})>"
