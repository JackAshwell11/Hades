"""Contains the functionality that manages the player menu and its events."""

from __future__ import annotations

# Custom
from hades.player.controller import PlayerController
from hades.player.view import PlayerView
from hades.view import BaseView

__all__ = ("Player",)


class Player(BaseView):
    """Manages the player menu and its events.

    Attributes:
        view: The renderer for the player.
        controller: The controller managing the player logic.
    """

    __slots__ = ("controller", "view")

    def __init__(self: Player) -> None:
        """Initialise the object."""
        super().__init__()
        self.view: PlayerView = PlayerView(self.window)
        self.controller: PlayerController = PlayerController(self.model, self.view)

    def setup(self: Player) -> None:
        """Set up the player view."""
        self.view.setup()
        self.controller.setup()

    def on_show_view(self: Player) -> None:
        """Process show view functionality."""
        self.controller.show_view()

    def on_hide_view(self: Player) -> None:
        """Process hide view functionality."""
        self.controller.hide_view()

    def on_draw(self: Player) -> None:
        """Render the screen."""
        self.view.draw()
