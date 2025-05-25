"""Contains the functionality that manages the player menu."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import View

# Custom
from hades.player.controller import PlayerController
from hades.player.view import PlayerView

if TYPE_CHECKING:
    from hades.model import GameModel

__all__ = ("Player",)


class Player(View):
    """Manages the game and its events.

    Attributes:
        view: The renderer for the player.
        controller: The controller managing the player logic.
    """

    __slots__ = ("controller", "view")

    def __init__(self: Player) -> None:
        """Initialise the object."""
        super().__init__()
        self.view: PlayerView = PlayerView(self.window)
        self.controller: PlayerController = PlayerController(self.view)

    def setup(self: Player, model: GameModel) -> None:
        """Set up the player view.

        Args:
            model: The model managing the player state.
        """
        self.controller.model = model
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
