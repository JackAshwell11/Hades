"""Represents the base class for all MVC views in the game."""

from __future__ import annotations

# Builtin
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Pip
from arcade.gui import UIManager

if TYPE_CHECKING:
    from hades.window import HadesWindow

__all__ = ("BaseView",)


class BaseView(ABC):
    """The base class for all MVC views in the game.

    Attributes:
        ui: The UI manager for the view.
    """

    __slots__ = ("ui", "window")

    @abstractmethod
    def _setup_layout(self: BaseView) -> None:
        """Set up the layout for the view.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def __init__(self: BaseView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the view.
        """
        self.window: HadesWindow = window
        self.ui: UIManager = UIManager()
        self._setup_layout()

    def draw(self: BaseView) -> None:
        """Draw the view elements."""
        self.window.clear()
        self.ui.draw()
