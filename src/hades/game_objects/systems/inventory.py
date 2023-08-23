"""Manages the inventory system and its various functionalities."""
from __future__ import annotations

# Custom
from hades.game_objects.base import SystemBase
from hades.game_objects.components import Inventory

__all__ = ("InventorySpaceError", "InventorySystem")


class InventorySpaceError(Exception):
    """Raised when there is a space problem with the inventory."""

    def __init__(self: InventorySpaceError, *, full: bool) -> None:
        """Initialise the object.

        Args:
            full: Whether the inventory is empty or full.
        """
        super().__init__(f"The inventory is {'full' if full else 'empty'}.")


class InventorySystem(SystemBase):
    """Provides facilities to manipulate inventory components."""

    def add_item_to_inventory(
        self: InventorySystem,
        game_object_id: int,
        item: int,
    ) -> None:
        """Add an item to the inventory.

        Args:
            game_object_id: The ID of the game object to add the item to.
            item: The item to add to the inventory.

        Raises:
            InventorySpaceError: The inventory is full.
        """
        inventory = self.registry.get_component_for_game_object(
            game_object_id,
            Inventory,
        )
        if len(inventory.inventory) == inventory.width * inventory.height:
            raise InventorySpaceError(full=True)
        inventory.inventory.append(item)

    def remove_item_from_inventory(
        self: InventorySystem,
        game_object_id: int,
        index: int,
    ) -> int:
        """Remove an item at a specific index.

        Args:
            game_object_id: The ID of the game object to remove an item from.
            index: The index to remove an item at.

        Returns:
            The item at position `index` in the inventory.

        Raises:
            InventorySpaceError: The inventory is empty.
        """
        inventory = self.registry.get_component_for_game_object(
            game_object_id,
            Inventory,
        )
        if len(inventory.inventory) < index:
            raise InventorySpaceError(full=False)
        return inventory.inventory.pop(index)
