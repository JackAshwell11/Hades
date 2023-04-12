"""Manages the different components available to the game objects."""
from __future__ import annotations

# Builtin
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hades.game_objects.enums import InventoryData
    from hades.game_objects.objects import GameObject

__all__ = ("ActionableMixin", "CollectibleMixin", "Inventory")


class Inventory:
    """Allow a game object to have a fixed size inventory."""

    __slots__ = (
        "inventory_data",
        "inventory",
    )

    def __init__(self: type[Inventory], inventory_data: InventoryData) -> None:
        """Initialise the object.

        Parameters
        ----------
        inventory_data: InventoryData
            The data for the inventory component.
        """
        self.inventory_data: InventoryData = inventory_data
        self.inventory: list[int] = []

    def add_item_to_inventory(self: type[Inventory], item: int) -> None:
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
        if (
            len(self.inventory)
            == self.inventory_data.width * self.inventory_data.height
        ):
            raise ValueError("Not enough space in the inventory")
        self.inventory.append(item)

    def remove_item_from_inventory(self: type[Inventory], index: int) -> int:
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
        if len(self.inventory) < index:
            raise ValueError("Not enough space in the inventory")
        return self.inventory.pop(index)


class ActionableMixin(metaclass=ABCMeta):
    """Allow a game object to have an action when interacted with."""

    @abstractmethod
    def action_use(self: type[GameObject]) -> None:
        """Process the game object's action."""
        raise NotImplementedError


class CollectibleMixin(metaclass=ABCMeta):
    """Allow a game object to be collected when interacted with."""

    @abstractmethod
    def collect_use(self: type[GameObject]) -> None:
        """Process the game object's collection."""
        raise NotImplementedError
