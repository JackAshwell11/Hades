"""Handles loading and storage of textures needed by the game."""

from __future__ import annotations

# Custom
from hades_extensions.game_objects import SPRITE_SIZE

__all__ = (
    "BiggerThanError",
    "grid_pos_to_pixel",
)


class BiggerThanError(Exception):
    """Raised when a value is less than a required value."""

    def __init__(self: BiggerThanError, *, min_value: float) -> None:
        """Initialise the object.

        Args:
            min_value: The minimum value that is allowed.
        """
        super().__init__(f"The input must be bigger than or equal to {min_value}.")


def grid_pos_to_pixel(x: int, y: int) -> tuple[float, float]:
    """Calculate the x and y position based on the dungeon or vector field position.

    Args:
        x: The x position in the dungeon or vector field.
        y: The x position in the dungeon or vector field.

    Returns:
        The x and y position of a sprite on the screen.

    Raises:
        BiggerThanError: The input must be bigger than or equal to 0.
    """
    # Check if the inputs are negative
    if x < 0 or y < 0:
        raise BiggerThanError(min_value=0)

    # Calculate the position on screen
    return (
        x * SPRITE_SIZE + SPRITE_SIZE / 2,
        y * SPRITE_SIZE + SPRITE_SIZE / 2,
    )
