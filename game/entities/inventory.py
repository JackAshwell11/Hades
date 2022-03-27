from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from constants.general import INVENTORY_HEIGHT, INVENTORY_WIDTH

if TYPE_CHECKING:
    from entities.base import Item
    from entities.player import Player


class Inventory:
    """
    Represents an inventory in the game which the player can store items in.

    Parameters
    ----------
    owner: Player
        The owner of this inventory.

    Attributes
    ----------
    width: int
        The width of the inventory.
    height: int
        The height of the inventory.
    array: list[Item]
        The list which stores the player's inventory.
    """

    def __init__(self, owner: Player) -> None:
        self.player: Player = owner
        self.width: int = INVENTORY_WIDTH
        self.height: int = INVENTORY_HEIGHT
        self.max_size = self.width * self.height
        self.array: list[Item] = []

    def __repr__(self) -> str:
        return (
            f"<Inventory (Width={self.width}) (Height={self.height}) (Max"
            f" size={self.max_size})>"
        )

    def add_item(self, item: Item) -> None:
        """
        Adds an item to the player's inventory.

        Parameters
        ----------
        item: Item
            The item to add to the player's inventory.

        Raises
        -------
        IndexError
            The player's inventory is full.
        """
        # Check if the array is full
        if len(self.array) == self.max_size:
            raise IndexError("Inventory is full.")

        # Add the item to the array
        self.array.append(item)
