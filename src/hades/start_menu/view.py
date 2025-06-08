"""Manages the rendering of start menu elements on the screen."""

from __future__ import annotations

# Builtin
from functools import partial
from typing import TYPE_CHECKING, Final

# Pip
from arcade import get_window
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIFlatButton, UIManager

if TYPE_CHECKING:
    from hades.window import HadesWindow

__all__ = (
    "QuitButton",
    "StartButton",
    "StartMenuView",
)

# The width of the start menu buttons
START_MENU_BUTTON_WIDTH: Final[int] = 200

# The spacing between the start menu widgets
START_MENU_WIDGET_SPACING: Final[int] = 20


class StartButton(UIFlatButton):
    """Represents a button that starts the game when clicked."""

    def __init__(self: StartButton) -> None:
        """Initialise the object."""
        super().__init__(text="Start Game", width=START_MENU_BUTTON_WIDTH)
        self.on_click = partial(  # type: ignore[method-assign]
            get_window().dispatch_event,
            "on_start_game",
        )


class QuitButton(UIFlatButton):
    """Represents a button that quits the game when clicked."""

    def __init__(self: QuitButton) -> None:
        """Initialise the object."""
        super().__init__(text="Quit Game", width=START_MENU_BUTTON_WIDTH)
        self.on_click = partial(  # type: ignore[method-assign]
            get_window().dispatch_event,
            "on_quit_game",
        )


class StartMenuView:
    """Manages the rendering of start menu elements on the screen.

    Attributes:
        ui: The UI manager for the start menu view.
    """

    __slots__ = ("ui", "window")

    def __init__(self: StartMenuView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the start menu.
        """
        self.window: HadesWindow = window
        self.ui: UIManager = UIManager()

        vertical_box = UIBoxLayout(space_between=START_MENU_WIDGET_SPACING)
        vertical_box.add(StartButton())
        vertical_box.add(QuitButton())
        self.ui.add(UIAnchorLayout(children=(vertical_box,)))

    def draw(self: StartMenuView) -> None:
        """Draw the start menu elements."""
        self.window.clear()
        self.ui.draw()
