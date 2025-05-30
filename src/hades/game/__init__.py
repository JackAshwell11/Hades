"""Contains the functionality that allows the user to play the game."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import View

# Custom
from hades.game.controller import GameController
from hades.game.view import GameView

if TYPE_CHECKING:
    from hades.model import HadesModel

__all__ = ("Game",)


class Game(View):
    """Manages the game and its events.

    Attributes:
        model: The model providing access to the game engine and its functionality.
        view: The renderer for the game.
        controller: The controller managing the game logic.
    """

    __slots__ = ("controller", "model", "view")

    def __init__(self: Game) -> None:
        """Initialise the object."""
        super().__init__()
        self.model: HadesModel = self.window.model
        self.view: GameView = GameView(self.window)
        self.controller: GameController = GameController(self.model, self.view)

    def setup(self: Game) -> None:
        """Set up the game."""
        self.view.setup()
        self.controller.setup()

    def on_show_view(self: Game) -> None:
        """Process show view functionality."""
        self.controller.show_view()

    def on_hide_view(self: Game) -> None:
        """Process hide view functionality."""
        self.controller.hide_view()

    def on_draw(self: Game) -> None:
        """Render the screen."""
        self.view.draw()

    def on_update(self: Game, delta_time: float) -> None:
        """Process game logic.

        Args:
            delta_time: The time since the last update.
        """
        self.controller.update(delta_time)

    def on_key_release(self: Game, symbol: int, modifiers: int) -> None:
        """Process key release functionality.

        Args:
            symbol: The key that was hit.
            modifiers: The bitwise AND of all modifiers (shift, ctrl, num lock) pressed
                during this event.
        """
        self.controller.key_release(symbol, modifiers)

    def on_mouse_motion(self: Game, x: int, y: int, dx: int, dy: int) -> None:
        """Process mouse motion functionality.

        Args:
            x: The x position of the mouse.
            y: The y position of the mouse.
            dx: The change in the x position of the mouse.
            dy: The change in the y position of the mouse.
        """
        self.controller.mouse_motion(x, y, dx, dy)
