"""Contains the functionality that manages the game options and its events."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.scenes.base import BaseScene
from hades.scenes.game_options.view import GameOptionsView

if TYPE_CHECKING:
    from typing import ClassVar

    from hades_engine import DifficultyLevel

__all__ = ("GameOptionsScene",)


class GameOptionsScene(BaseScene[GameOptionsView]):
    """Manages the game options and its events."""

    # The view type for the scene
    _view_type: ClassVar[type[GameOptionsView]] = GameOptionsView

    def add_callbacks(self: GameOptionsScene) -> None:
        """Set up the controller callbacks."""

    def on_start_level(self: GameOptionsScene) -> None:
        """Process the start level functionality."""
        self.model.game_engine.enter_dungeon()

    def on_difficulty_change(
        self: GameOptionsScene,
        difficulty_level: DifficultyLevel,
    ) -> None:
        """Process the difficulty change functionality.

        Args:
            difficulty_level: The difficulty level to set.
        """
        self.model.game_state.difficulty_level = difficulty_level
        for level, difficulty_button in self.view.difficulty_layout.buttons.items():
            difficulty_button.clicked = level == difficulty_level
        self.view.difficulty_layout.trigger_render()
