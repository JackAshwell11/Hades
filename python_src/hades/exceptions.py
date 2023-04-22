"""Stores the different exceptions that can be raised by the game."""
from __future__ import annotations

__all__ = ("BiggerThanError", "SpaceError")


class SpaceError(Exception):
    """Raised when there is not enough room in a container."""

    def __init__(self: SpaceError, name: str) -> None:
        """Initialise the object.

        Parameters
        ----------
        name: str
            The name of the container that does not have enough room.
        """
        super().__init__(f"The `{name}` container does not have enough room.")


class BiggerThanError(Exception):
    """Raised when a value is less than a required value."""

    def __init__(self: BiggerThanError, min_value: float) -> None:
        """Initialise the object.

        Parameters
        ----------
        min_value: float
            The minimum value that is allowed.
        """
        super().__init__(f"The input must be bigger than or equal to {min_value}.")
