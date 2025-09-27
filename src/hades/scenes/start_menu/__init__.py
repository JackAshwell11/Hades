"""Contains the functionality that manages the start menu and its events."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades import SceneType
from hades.scenes.base import BaseScene
from hades.scenes.start_menu.view import StartMenuView
from hades_engine import EventType, SaveFileInfo, add_callback

if TYPE_CHECKING:
    from typing import ClassVar

    from arcade.gui import UIOnClickEvent

__all__ = ("StartMenuScene",)


class StartMenuScene(BaseScene[StartMenuView]):
    """Manages the start menu and its events."""

    # The view type for the scene
    _view_type: ClassVar[type[StartMenuView]] = StartMenuView

    def add_callbacks(self: StartMenuScene) -> None:
        """Add callbacks for the scene."""
        add_callback(EventType.SaveFilesUpdated, self.on_save_files_updated)

    def on_new_game(self: StartMenuScene, _: UIOnClickEvent) -> None:
        """Process new game functionality."""
        self.model.save_manager.new_game()

    def on_load_game(self: StartMenuScene, _: UIOnClickEvent) -> None:
        """Process load game functionality."""
        self.view.window.show_view(self.view.window.scenes[SceneType.LOAD_GAME])

    def on_quit_game(self: StartMenuScene, _: UIOnClickEvent) -> None:
        """Process quit game functionality."""
        self.view.window.close()

    def on_save_files_updated(self: StartMenuScene, saves: list[SaveFileInfo]) -> None:
        """Process save files updated functionality.

        Args:
            saves: The list of save files that have been updated.
        """
        if len(saves) == 0:
            self.view.set_load_game_button_state(disabled=True)
        else:
            self.view.set_load_game_button_state(disabled=False)
