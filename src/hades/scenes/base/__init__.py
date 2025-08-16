"""Represents the base class for all scenes in the game."""

from __future__ import annotations

# Builtin
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

# Pip
from arcade import View

if TYPE_CHECKING:
    from typing import ClassVar

    from hades.model import HadesModel
    from hades.scenes.base.view import BaseView


class BaseScene[V: BaseView](ABC, View):
    """The base class for all scenes in the game.

    Attributes:
        model: The model providing access to the game engine and its functionality.
        view: The renderer for the scene.
    """

    # The view type for the scene
    _view_type: ClassVar[type[BaseView]] = None  # type: ignore[assignment]

    __slots__ = ("__weakref__", "view")

    def __init__(self: BaseScene[V]) -> None:
        """Initialise the object."""
        super().__init__()
        if self._view_type is None:
            error = "Subclasses must override _view_type"  # type: ignore[unreachable]
            raise NotImplementedError(error)
        self.model: HadesModel = self.window.model
        self.view: V = cast("V", self._view_type(self.window))
        self.add_callbacks()

    @abstractmethod
    def add_callbacks(self: BaseScene[V]) -> None:
        """Add callbacks for the scene.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def on_show_view(self: BaseScene[V]) -> None:
        """Process show view functionality."""
        self.view.ui.enable()
        self.view.window.push_handlers(self)

    def on_hide_view(self: BaseScene[V]) -> None:
        """Process hide view functionality."""
        self.view.window.save_background()
        self.view.ui.disable()
        self.view.window.remove_handlers(self)

    def on_draw(self: BaseScene[V]) -> None:
        """Render the screen."""
        self.view.draw()
