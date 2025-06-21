"""Manages the rendering of shop elements on the screen."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIFlatButton, UIManager

# Custom
from hades import UI_PADDING, ViewType
from hades.constructors import IconType
from hades.grid_layout import ItemButton, PaginatedGridLayout

if TYPE_CHECKING:
    from arcade import Texture

    from hades.window import HadesWindow

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


class ShopView:
    """Manages the rendering of shop elements on the screen.

    Attributes:
        ui: The UI manager for the shop view.
        grid_layout: The layout for displaying the shop's inventory items.
    """

    __slots__ = ("grid_layout", "ui", "window")

    def __init__(self: ShopView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the start menu.
        """
        self.window: HadesWindow = window
        self.ui: UIManager = UIManager()
        self.grid_layout: PaginatedGridLayout[ShopItemButton] = PaginatedGridLayout()

        self.ui.add(self.window.background_image)
        layout = UIBoxLayout(vertical=True, space_between=UI_PADDING)
        layout.add(self.grid_layout)
        back_button = UIFlatButton(text="Back")
        back_button.on_click = (  # type: ignore[method-assign]
            lambda _: self.window.show_view(  # type: ignore[assignment]
                self.window.views[ViewType.GAME],
            )
        )
        layout.add(back_button)
        self.ui.add(UIAnchorLayout(children=(layout,)))

    def draw(self: ShopView) -> None:
        """Draw the shop elements."""
        self.window.clear()
        self.ui.draw()
