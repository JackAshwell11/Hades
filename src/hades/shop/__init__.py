"""Contains the functionality that manages the shop menu and its events."""

from __future__ import annotations

# Custom
from hades.shop.controller import ShopController
from hades.shop.view import ShopView
from hades.view import BaseView

__all__ = ("Shop",)


class Shop(BaseView):
    """Manages the shop menu and its events.

    Attributes:
        view: The renderer for the shop.
        controller: The controller managing the shop logic.
    """

    __slots__ = ("controller", "view")

    def __init__(self: Shop) -> None:
        """Initialise the object."""
        super().__init__()
        self.view: ShopView = ShopView(self.window)
        self.controller: ShopController = ShopController(self.model, self.view)

    def add_callbacks(self: Shop) -> None:
        """Add the callbacks for the shop."""
        self.controller.add_callbacks()

    def on_show_view(self: Shop) -> None:
        """Process show view functionality."""
        self.controller.show_view()

    def on_hide_view(self: Shop) -> None:
        """Process hide view functionality."""
        self.controller.hide_view()

    def on_draw(self: Shop) -> None:
        """Render the screen."""
        self.view.draw()
