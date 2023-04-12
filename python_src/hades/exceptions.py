"""Finds enemies within melee range of the player using a ray-casting shader program."""
from __future__ import annotations

__all__ = ("BiggerThanError", "SpaceError")


class BiggerThanError(Exception):
    """Raised when a value is less than a required value."""

    def __init__(self: BiggerThanError, min_value: float) -> None:
        """Initialise the object.

        Parameters
        ----------
        min_value: float
            The minimum value that is allowed.
        """
        super().__init__(f"The input must be bigger than or equal to {min_value}")


class SpaceError(Exception):
    """Raised when there is not enough room in a container."""

    def __init__(self: SpaceError, name: str) -> None:
        """Initialise the object.

        Parameters
        ----------
        name: str
            The name of the container that does not have enough room.
        """
        super().__init__(f"The `{name}` container does not have enough room")
