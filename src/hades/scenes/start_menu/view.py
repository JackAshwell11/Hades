"""Manages the rendering of start menu elements on the screen."""

from __future__ import annotations

# Builtin
from functools import partial
from typing import TYPE_CHECKING, Final

# Pip
from arcade import get_window
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIFlatButton

# Custom
from hades import BUTTON_WIDTH, MENU_WIDGET_SPACING
from hades.scenes.base.view import BaseView

if TYPE_CHECKING:
    from hades.window import HadesWindow

__all__ = ("LoadGameButton", "NewGameButton", "StartMenuView")

# The width of the start menu buttons
START_MENU_BUTTON_WIDTH: Final[int] = 200


class NewGameButton(UIFlatButton):
    """Represents a button that starts a new game when clicked."""

    def __init__(self: NewGameButton) -> None:
        """Initialise the object."""
        super().__init__(text="New Game", width=START_MENU_BUTTON_WIDTH)
        self.on_click = partial(  # type: ignore[method-assign]
            get_window().dispatch_event,
            "on_new_game",
        )


class LoadGameButton(UIFlatButton):
    """Represents a button that loads a game when clicked."""

    def __init__(self: LoadGameButton) -> None:
        """Initialise the object."""
        super().__init__(width=START_MENU_BUTTON_WIDTH)
        self.on_click = partial(  # type: ignore[method-assign]
            get_window().dispatch_event,
            "on_load_game",
        )


class QuitButton(UIFlatButton):
    """Represents a button that quits the game when clicked."""

    def __init__(self: QuitButton) -> None:
        """Initialise the object."""
        super().__init__(text="Quit Game", width=BUTTON_WIDTH)
        self.on_click = partial(  # type: ignore[method-assign]
            get_window().dispatch_event,
            "on_quit_game",
        )


class StartMenuView(BaseView):
    """Manages the rendering of start menu elements on the screen."""

    __slots__ = ("load_game_button",)

    def _setup_layout(self: StartMenuView) -> None:
        """Set up the layout for the view."""
        vertical_box = UIBoxLayout(space_between=MENU_WIDGET_SPACING)
        vertical_box.add(NewGameButton())
        vertical_box.add(self.load_game_button)
        vertical_box.add(QuitButton())
        self.ui.add(UIAnchorLayout(children=(vertical_box,)))

    def __init__(self: StartMenuView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the game.
        """
        self.load_game_button: LoadGameButton = LoadGameButton()
        super().__init__(window)

    def set_load_game_button_state(self: StartMenuView, *, disabled: bool) -> None:
        """Set the state of the load game button.

        Args:
            disabled: Whether the button should be disabled or not.
        """
        self.load_game_button.disabled = disabled
        if disabled:
            self.load_game_button.text = "No Saves Available"
        else:
            self.load_game_button.text = "Load Game"
