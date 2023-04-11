"""Manages the different components available to the game objects."""
from __future__ import annotations

# Builtin
from abc import ABCMeta, abstractmethod

__all__ = ("Actionable", "Collectible", "Inventory")


class Inventory:
    """Allow a game object to have a fixed size inventory.

    Attributes
    ----------
    inventory_width: int
        The width of the inventory.
    inventory_height: int
        The height of the inventory.
    """

    __slots__ = (
        "inventory_width",
        "inventory_height",
        "inventory",
    )

    def __init__(self, width: int, height: int) -> None:
        self.inventory_width: int = width
        self.inventory_height: int = height
        self.inventory: list[int] = []

    @property
    def inventory_size(self) -> int:
        """Get the current size of the inventory.

        Returns
        -------
        int
            The current size of the inventory.
        """
        return len(self.inventory)

    def add_item_to_inventory(self, item: int) -> None:
        """Add an item to the inventory.

        Parameters
        ----------
        item: int
            The item to add to the inventory.

        Raises
        ------
        ValueError
            Not enough space in the inventory
        """
        if self.inventory_size == self.inventory_width * self.inventory_height:
            raise ValueError("Not enough space in the inventory")
        self.inventory.append(item)

    def remove_item_from_inventory(self, index: int) -> int:
        """Remove an item at a specific index.

        Parameters
        ----------
        index: int
            The index to remove an item at.

        Returns
        -------
        ValueError
            Not enough space in the inventory
        """
        if self.inventory_size < index:
            raise ValueError("Not enough space in the inventory")
        return self.inventory.pop(index)


class Actionable(metaclass=ABCMeta):
    """Allow a game object to have an action when interacted with."""

    @abstractmethod
    def action_use(self) -> None:
        """Process the game object's action."""
        raise NotImplementedError


class Collectible(metaclass=ABCMeta):
    """Allow a game object to be collected when interacted with."""

    @abstractmethod
    def collect_use(self) -> None:
        """Process the game object's collection."""
        raise NotImplementedError
