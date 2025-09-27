"""Contains the functionality that manages the load game menu and its events."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.scenes.base import BaseScene
from hades.scenes.load_game.view import LoadGameMenuView, SaveEntry
from hades_engine import EventType, SaveFileInfo, add_callback

if TYPE_CHECKING:
    from typing import ClassVar

    from arcade.gui import UIOnClickEvent

__all__ = ("LoadGameMenuScene",)


class LoadGameMenuScene(BaseScene[LoadGameMenuView]):
    """Manages the start menu and its events."""

    # The view type for the scene
    _view_type: ClassVar[type[LoadGameMenuView]] = LoadGameMenuView

    def add_callbacks(self: LoadGameMenuScene) -> None:
        """Add callbacks for the scene."""
        add_callback(EventType.SaveFilesUpdated, self.on_save_files_updated)

    def on_save_files_updated(
        self: LoadGameMenuScene,
        saves: list[SaveFileInfo],
    ) -> None:
        """Process save files updated functionality.

        Args:
            saves: The list of save files that have been updated.
        """
        self.view.save_layout.clear()
        for index, save in enumerate(saves):
            self.view.save_layout.add(SaveEntry(save, index))

    def on_load_save(
        self: LoadGameMenuScene,
        save_index: int,
        _: UIOnClickEvent,
    ) -> None:
        """Process load save functionality.

        Args:
            save_index: The index of the save file to load.
        """
        self.model.save_manager.load_save(save_index)

    def on_delete_save(
        self: LoadGameMenuScene,
        save_index: int,
        _: UIOnClickEvent,
    ) -> None:
        """Process delete save functionality.

        Args:
            save_index: The index of the save file to delete.
        """
        self.model.save_manager.delete_save(save_index)
