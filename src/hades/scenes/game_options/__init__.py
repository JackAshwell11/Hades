"""Contains the functionality that manages the game options and its events."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade.key import E

# Custom
from hades.scenes.base import BaseScene
from hades.scenes.game_options.view import GameOptionsView
from hades_engine import EventType, add_callback

if TYPE_CHECKING:
    from typing import ClassVar

__all__ = ("GameOptionsScene",)


class GameOptionsScene(BaseScene[GameOptionsView]):
    """Manages the game options and its events."""

    # The view type for the scene
    _view_type: ClassVar[type[GameOptionsView]] = GameOptionsView

    def add_callbacks(self: GameOptionsScene) -> None:
        """Set up the controller callbacks."""
        add_callback(EventType.GameOptionsOpen, self.on_game_options_open)

    def on_optioned_start_level(self: GameOptionsScene, seed: str) -> None:
        """Process the start level functionality with options."""
        if seed:
            self.model.game_state.set_seed(seed)
        self.model.input_handler.on_key_release(E, 0)

    def on_game_options_open(self: GameOptionsScene) -> None:
        """Process game options open functionality."""
        self.view.window.show_view(self)
