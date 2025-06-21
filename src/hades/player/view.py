"""Manages the rendering of player elements on the screen."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIFlatButton, UIManager

# Custom
from hades import UI_PADDING, ViewType
from hades.grid_layout import ItemButton, PaginatedGridLayout

if TYPE_CHECKING:
    from arcade import Texture

    from hades.sprite import HadesSprite
    from hades.window import HadesWindow

__all__ = ("InventoryItemButton", "PlayerView")


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


class PlayerView:
    """Manages the rendering of player elements on the screen.

    Attributes:
        ui: The UI manager for the player view.
        grid_layout: The layout for displaying the player's inventory items.
    """

    __slots__ = ("grid_layout", "ui", "window")

    def __init__(self: PlayerView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the start menu.
        """
        self.window: HadesWindow = window
        self.ui: UIManager = UIManager()
        self.grid_layout: PaginatedGridLayout[InventoryItemButton] = (
            PaginatedGridLayout()
        )

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

    def draw(self: PlayerView) -> None:
        """Draw the player elements."""
        self.window.clear()
        self.ui.draw()
