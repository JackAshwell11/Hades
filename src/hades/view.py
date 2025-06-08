"""Represents the base view for the game."""

from __future__ import annotations

# Builtin
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Pip
from arcade import View

if TYPE_CHECKING:
    from hades.model import HadesModel

__all__ = ("BaseView",)


class BaseView(ABC, View):
    """Base class for all views in the game.

    Attributes:
        model: The model providing access to the game engine and its functionality.
    """

    __slots__ = ("model",)

    def __init__(self: BaseView) -> None:
        """Initialise the object."""
        super().__init__()
        self.model: HadesModel = self.window.model

    @abstractmethod
    def add_callbacks(self: BaseView) -> None:
        """Add callbacks for the view.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
