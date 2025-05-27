"""Represents a distinct state in the game."""

from __future__ import annotations

# Builtin
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygame.event import Event
    from pygame.surface import Surface

    from hades.window import HadesWindow

__all__ = ("Scene",)


class Scene(ABC):
    """Represents a distinct state in the game.

    Attributes:
        window: The window where the scene is displayed.
    """

    def __init__(self: Scene, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window where the scene is displayed.
        """
        self.window: HadesWindow = window

    @abstractmethod
    def handle_events(self: Scene, events: list[Event]) -> None:
        """Handle events for the scene.

        Args:
            events: A list of events to handle.
        """
        raise NotImplementedError

    @abstractmethod
    def draw(self: Scene, surface: Surface) -> None:
        """Draw the scene.

        Args:
            surface: The surface to draw on.
        """
        raise NotImplementedError
