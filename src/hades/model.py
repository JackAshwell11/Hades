"""Provides access to the game engine and its functionality."""

from __future__ import annotations

# Builtin
from functools import cached_property
from typing import TYPE_CHECKING

# Custom
from hades_extensions import GameEngine

if TYPE_CHECKING:
    from hades_extensions import GameState, InputHandler, SaveManager
    from hades_extensions.ecs import Registry

__all__ = ("HadesModel",)


class HadesModel:
    """Provides access to the game engine and its functionality.

    Attributes:
        game_engine: The engine which manages the registry and events.
    """

    __slots__ = ("__dict__", "game_engine")

    def __init__(self: HadesModel) -> None:
        """Initialise the object."""
        self.game_engine: GameEngine = GameEngine()

    @property
    def registry(self: HadesModel) -> Registry:
        """Get the registry which manages the game objects, components, and systems.

        Returns:
            The registry which manages the game objects, components, and systems.
        """
        return self.game_engine.registry

    @cached_property
    def game_state(self: HadesModel) -> GameState:
        """Get the game state which stores the state of the game.

        Returns:
            The game state which stores the state of the game.
        """
        return self.game_engine.game_state

    @cached_property
    def input_handler(self: HadesModel) -> InputHandler:
        """Get the input handler which handles input events.

        Returns:
            The input handler which handles input events.
        """
        return self.game_engine.input_handler

    @cached_property
    def save_manager(self: HadesModel) -> SaveManager:
        """Get the save manager which manages the saving and loading of game states.

        Returns:
            The save manager which manages the saving and loading of game states.
        """
        return self.game_engine.save_manager

    @property
    def player_id(self: HadesModel) -> int:
        """Get the ID of the player game object.

        Returns:
            The ID of the player game object.
        """
        return self.game_state.player_id
