from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from constants import INVENTORY_SIZE, TileType

if TYPE_CHECKING:
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
    size: int
        The size of the inventory.
    array: list[TileType]
        The list which stores the player's inventory.
    """

    def __init__(self, owner: Player) -> None:
        self.owner: Player = owner
        self.size: int = INVENTORY_SIZE
        self.array: list[TileType] = []

    def __repr__(self) -> str:
        return f"<Inventory (Size={self.size})>"

    def add_item(self, item: TileType) -> None:
        """
        Adds an item to the player's inventory.

        Parameters
        ----------
        item: TileType
            The item to add to the player's inventory.

        Raises
        -------
        IndexError
            The player's inventory is full.
        """
        # Check if the array is full
        if len(self.array) == self.size:
            raise IndexError("Inventory is full.")

        # Add the item to the array
        self.array.append(item)
