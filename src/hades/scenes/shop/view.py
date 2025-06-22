"""Manages the rendering of shop elements on the screen."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.constructors import IconType
from hades.grid_layout import GridView, ItemButton

if TYPE_CHECKING:
    from arcade import Texture

__all__ = ("ShopItemButton", "ShopView")


class ShopItemButton(ItemButton):
    """Represents a shop item button."""

    __slots__ = ("cost", "description", "name", "shop_index")

    def __init__(
        self: ShopItemButton,
        index: int,
        data: tuple[str, str, str],
        cost: int,
    ) -> None:
        """Initialise the object.

        Args:
            index: The index of the item in the shop.
            data: A tuple containing the name, description, and icon type of the item.
            cost: The cost of the item.
        """
        super().__init__("Buy")
        self.shop_index: int = index
        self.name: str = data[0]
        self.description: str = data[1]
        self.cost: int = cost
        self.texture = IconType[data[2].upper()].get_texture()

    def get_info(self: ShopItemButton) -> tuple[str, str, Texture]:
        """Get the information about the item.

        Returns:
            The name, description, and texture of the item.
        """
        return self.name, f"{self.description}\nCost: {self.cost}", self.texture


class ShopView(GridView[ShopItemButton]):
    """Manages the rendering of shop elements on the screen."""
