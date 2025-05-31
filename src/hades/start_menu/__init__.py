"""Contains the functionality that manages the start menu and its events."""

from __future__ import annotations

# Custom
from hades.start_menu.controller import StartMenuController
from hades.start_menu.view import StartMenuView
from hades.view import BaseView

__all__ = ("StartMenu",)


class StartMenu(BaseView):
    """Manages the start menu and its events.

    Attributes:
        view: The renderer for the start menu.
        controller: The controller managing the start menu logic.
    """

    __slots__ = ("controller", "view")

    def __init__(self: StartMenu) -> None:
        """Initialise the object."""
        super().__init__()
        self.view: StartMenuView = StartMenuView(self.window)
        self.controller: StartMenuController = StartMenuController(
            self.model,
            self.view,
        )

    def setup(self: StartMenu) -> None:
        """Set up the start menu view."""
        self.view.setup()
        self.controller.setup()

    def on_show_view(self: StartMenu) -> None:
        """Process show view functionality."""
        self.controller.show_view()

    def on_hide_view(self: StartMenu) -> None:
        """Process hide view functionality."""
        self.controller.hide_view()

    def on_draw(self: StartMenu) -> None:
        """Render the screen."""
        self.view.draw()
