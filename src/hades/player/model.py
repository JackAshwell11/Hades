"""Manages the player state and its logic."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from hades_extensions.ecs import Registry

__all__ = ("PlayerModel",)


class PlayerModel:
    """Manages the player state and its logic.

    Attributes:
        registry: The registry which manages the game objects, components, and systems.
        player_id: The ID of the player game object.
    """

    __slots__ = ("player_id", "registry")

    def __init__(self: PlayerModel) -> None:
        """Initialise the object."""
        self.registry: Registry = cast("Registry", None)
        self.player_id: int = -1

    def setup(self: PlayerModel, registry: Registry, player_id: int) -> None:
        """Set up the player model.

        Args:
            registry: The registry which manages the game objects, components, and
                systems.
            player_id: The ID of the player game object.
        """
        self.registry = registry
        self.player_id = player_id
