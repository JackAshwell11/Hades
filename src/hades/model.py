"""Provides access to the game engine and its functionality."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades_extensions import GameEngine

if TYPE_CHECKING:
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

    @property
    def player_id(self: HadesModel) -> int:
        """Get the ID of the player game object.

        Returns:
            The ID of the player game object.
        """
        return self.game_engine.player_id
