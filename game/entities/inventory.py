from __future__ import annotations

# Builtin
import math
from typing import TYPE_CHECKING

# Custom
from constants import INVENTORY_SIZE

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
    size: int
        The size of the inventory.
    grid_size: int
        The size of a single column/row of the player's inventory.
    array: list[Item]
        The list which stores the player's inventory.
    """

    def __init__(self, owner: Player) -> None:
        self.player: Player = owner
        self.size: int = INVENTORY_SIZE
        self.grid_size: int = int(math.sqrt(self.size))
        self.array: list[Item] = []

    def __repr__(self) -> str:
        return f"<Inventory (Size={self.size})>"

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
        if len(self.array) == self.size:
            raise IndexError("Inventory is full.")

        # Add the item to the array
        self.array.append(item)

        # Check if we need to update the equipped consumable for the player
        self.set_next_consumable_index()

    def set_next_consumable_index(self) -> None:
        pass

    def set_previous_consumable_index(self) -> None:
        pass

    # def set_next_consumable_index(self) -> None:
    #     """Sets the player's currently equipped consumable index to the next
    #     consumable if there is one."""
    #
    #     # REDO THIS!
    #
    #     # Get the currently equipped consumable's index
    #     current_index = self.player.equipped_consumable
    #
    #     # Treat the array as a circular queue so loop back to the start when we've
    #     # reached the end
    #     for _ in range(len(self.array)):
    #         # Go to the next item
    #         current_index += 1
    #
    #         # Check if we need to loop back to the start
    #         if current_index == self.size - 1:
    #             current_index = 0
    #
    #         # Check if the current item is a consumable
    #         if self.array[current_index] in CONSUMABLES:
    #             # Exit the loop since we've found a consumable
    #             break
    #
    #     # Check if the new index is actually a consumable. This occurs when the player
    #     # uses all the consumables in the inventory, so we want to set it back to -1
    #     if self.array[current_index] in CONSUMABLES:
    #         self.player.equipped_consumable = current_index
    #     else:
    #         self.player.equipped_consumable = -1
