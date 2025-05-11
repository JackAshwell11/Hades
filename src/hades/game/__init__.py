"""Contains the functionality that allows the user to play the game."""

from __future__ import annotations

# Pip
from arcade import View

# Custom
from hades import ViewType
from hades.game.controller import GameController
from hades.game.model import GameModel
from hades.game.view import GameView
from hades.views.player import PlayerView

__all__ = ("Game",)


class Game(View):
    """Manages the game and its events.

    Attributes:
        model: The model managing the game state.
        view: The renderer for the game.
        controller: The controller managing the game logic.
    """

    __slots__ = ("controller", "model", "view")

    def __init__(self: Game) -> None:
        """Initialise the object."""
        super().__init__()
        self.model: GameModel = GameModel()
        self.view: GameView = GameView(self.window)
        self.controller: GameController = GameController(self.model, self.view)
        self.window.views[ViewType.PLAYER] = PlayerView()

    def setup(self: Game, level: int, seed: int | None = None) -> None:
        """Set up the game.

        Args:
            level: The level to create a game for.
            seed: The seed to use for the game engine.
        """
        self.model.setup(level, seed)
        self.view.setup()
        self.controller.setup()
        self.window.views[ViewType.PLAYER].setup(
            self.model.registry,
            self.model.player_id,
        )

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
