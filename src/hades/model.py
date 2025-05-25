"""Manages the game state and its logic."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast

# Pip
from arcade import unschedule

# Custom
from hades_extensions import GameEngine

if TYPE_CHECKING:
    from hades_extensions.ecs import Registry

__all__ = ("GameModel",)


class GameModel:
    """Manages the game state and logic.

    Attributes:
        game_engine: The engine which manages the registry and events.
        player_id: The ID of the player game object.
    """

    __slots__ = ("game_engine", "player_id")

    def __init__(self: GameModel) -> None:
        """Initialise the object."""
        self.game_engine: GameEngine = cast("GameEngine", None)
        self.player_id: int = -1

    def setup(self: GameModel, level: int, seed: int | None = None) -> None:
        """Set up the game model.

        Args:
            level: The level to create a game for.
            seed: The seed to use for the game engine.
        """
        if self.game_engine:
            unschedule(self.game_engine.generate_enemy)
        self.game_engine = GameEngine(level, seed)

    @property
    def registry(self: GameModel) -> Registry:
        """Get the registry which manages the game objects, components, and systems.

        Returns:
            The registry which manages the game objects, components, and systems.
        """
        return self.game_engine.get_registry()
