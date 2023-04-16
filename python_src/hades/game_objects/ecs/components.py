"""Manages various components available to the game objects."""
from __future__ import annotations

# Builtin
from abc import ABCMeta, abstractmethod

# Custom
from hades.exceptions import SpaceError

__all__ = ("Actionable", "Collectible", "InventoryMixin")


class InventoryMixin:
    """Allows a game object to have a fixed size inventory.

    Attributes
    ----------
    inventory: list[int]
        The game object's inventory.
    """

    __slots__ = (
        "width",
        "height",
        "inventory",
    )

    def __init__(self: InventoryMixin, width: int, height: int) -> None:
        """Initialise the object.

        Parameters
        ----------
        width: int
            The width of the inventory.
        height: int
            The height of the inventory.
        """
        self.width: int = width
        self.height: int = height
        self.inventory: list[int] = []

    def add_item_to_inventory(self: InventoryMixin, item: int) -> None:
        """Add an item to the inventory.

        Parameters
        ----------
        item: int
            The item to add to the inventory.

        Raises
        ------
        SpaceError
            The inventory container does not have enough room
        """
        if len(self.inventory) == self.width * self.height:
            raise SpaceError(self.__class__.__name__.lower())
        self.inventory.append(item)

    def remove_item_from_inventory(self: InventoryMixin, index: int) -> int:
        """Remove an item at a specific index.

        Parameters
        ----------
        index: int
            The index to remove an item at.


        Raises
        ------
        SpaceError
            The inventory container does not have enough room

        Returns
        -------
        ValueError
            Not enough space in the inventory
        """
        if len(self.inventory) < index:
            raise SpaceError(self.__class__.__name__.lower())
        return self.inventory.pop(index)


class Actionable(metaclass=ABCMeta):
    """Allows a game object to have an action when interacted with."""

    @abstractmethod
    def action_use(self: Actionable) -> None:
        """Process the game object's action."""
        raise NotImplementedError


class Collectible(metaclass=ABCMeta):
    """Allows a game object to be collected when interacted with."""

    @abstractmethod
    def collect_use(self: Collectible) -> None:
        """Process the game object's collection."""
        raise NotImplementedError
