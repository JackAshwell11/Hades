"""Contains the functionality that manages the start menu and its events."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades import SceneType
from hades.scenes.base import BaseScene
from hades.scenes.start_menu.view import StartMenuView

if TYPE_CHECKING:
    from typing import ClassVar

    from arcade.gui import UIOnClickEvent

__all__ = ("StartMenuScene",)


class StartMenuScene(BaseScene[StartMenuView]):
    """Manages the start menu and its events."""

    # The view type for the scene
    _view_type: ClassVar[type[StartMenuView]] = StartMenuView

    def add_callbacks(self: StartMenuScene) -> None:
        """Set up the controller callbacks."""
        self.view.window.register_event_type("on_start_game")
        self.view.window.register_event_type("on_quit_game")

    def on_start_game(self: StartMenuScene, _: UIOnClickEvent) -> None:
        """Process start game functionality."""
        self.view.window.show_view(self.view.window.scenes[SceneType.GAME])

    def on_quit_game(self: StartMenuScene, _: UIOnClickEvent) -> None:
        """Process quit game functionality."""
        self.view.window.close()
