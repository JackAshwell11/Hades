"""Manages the rendering of inventory elements on the screen."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.grid_layout import GridView, ItemButton

if TYPE_CHECKING:
    from arcade import Texture

    from hades.sprite import HadesSprite

__all__ = ("InventoryItemButton", "InventoryView")


class InventoryItemButton(ItemButton):
    """Represents an inventory item button."""

    __slots__ = ("sprite_object",)

    def __init__(self: InventoryItemButton, sprite_object: HadesSprite) -> None:
        """Initialise the object."""
        super().__init__("Use")
        self.sprite_object: HadesSprite = sprite_object
        self.texture = sprite_object.texture

    def get_info(self: InventoryItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Returns:
            The name, description, and texture of the item.
        """
        return self.sprite_object.name, self.sprite_object.description, self.texture


class InventoryView(GridView[InventoryItemButton]):
    """Manages the rendering of inventory elements on the screen."""
