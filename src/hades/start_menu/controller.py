"""Manages the start menu flow and registry callbacks."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades import ViewType

if TYPE_CHECKING:
    from arcade.gui import UIOnClickEvent

    from hades.model import HadesModel
    from hades.start_menu.view import StartMenuView

__all__ = ("StartMenuController",)


class StartMenuController:
    """Manages the start menu flow and registry callbacks."""

    __slots__ = ("__weakref__", "model", "view")

    def __init__(
        self: StartMenuController,
        model: HadesModel,
        view: StartMenuView,
    ) -> None:
        """Initialise the object.

        Args:
            model: The model providing access to the game engine and its functionality.
            view: The renderer for the StartMenu.
        """
        self.model: HadesModel = model
        self.view: StartMenuView = view

    def add_callbacks(self: StartMenuController) -> None:
        """Set up the controller callbacks."""
        self.view.window.register_event_type("on_start_game")
        self.view.window.register_event_type("on_quit_game")
        self.view.window.push_handlers(self)

    def show_view(self: StartMenuController) -> None:
        """Process show view functionality."""
        self.view.ui.enable()

    def hide_view(self: StartMenuController) -> None:
        """Process hide view functionality."""
        self.view.ui.disable()

    def on_start_game(self: StartMenuController, _: UIOnClickEvent) -> None:
        """Process start game functionality."""
        self.view.window.show_view(self.view.window.views[ViewType.GAME])

    def on_quit_game(self: StartMenuController, _: UIOnClickEvent) -> None:
        """Process quit game functionality."""
        self.view.window.close()
