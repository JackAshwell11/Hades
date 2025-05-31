"""Contains the functionality that manages the game and its events."""

from __future__ import annotations

# Custom
from hades.game.controller import GameController
from hades.game.view import GameView
from hades.view import BaseView

__all__ = ("Game",)


class Game(BaseView):
    """Manages the game and its events.

    Attributes:
        view: The renderer for the game.
        controller: The controller managing the game logic.
    """

    __slots__ = ("controller", "view")

    def __init__(self: Game) -> None:
        """Initialise the object."""
        super().__init__()
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
