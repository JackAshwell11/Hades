"""Manages the rendering of game option elements on the screen."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Final

# Pip
from arcade import color, get_window
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIFlatButton, UIInputText, UILabel
from arcade.gui.widgets.text import UIInputTextStyle  # type: ignore[attr-defined]

# Custom
from hades import UI_BACKGROUND_COLOUR, UI_PADDING, BackButton
from hades.scenes.base.view import BaseView

if TYPE_CHECKING:
    from hades.window import HadesWindow

__all__ = ("GameOptionsView",)

# The width of the input text for the seed
SEED_INPUT_WIDTH: Final[int] = 200


class OptionsPanel(UIBoxLayout):
    """Represents a panel for specifying game options."""

    __slots__ = ()

    def __init__(self: OptionsPanel) -> None:
        """Initialise the object."""
        super().__init__(space_between=UI_PADDING)
        horizontal_layout = UIBoxLayout(vertical=False, space_between=UI_PADDING)
        horizontal_layout.add(UILabel("Enter seed:", text_color=color.BLACK))
        seed_input = UIInputText(
            width=SEED_INPUT_WIDTH,
            text_color=color.BLACK,
            caret_color=color.BLACK,
            style={"normal": UIInputTextStyle(border=color.BLACK)},
        )
        horizontal_layout.add(seed_input)
        self.add(horizontal_layout)
        start_button = UIFlatButton(text="Start Game")
        start_button.on_click = (  # type: ignore[method-assign]
            lambda _: get_window().dispatch_event(  # type: ignore[assignment]
                "on_optioned_start_level",
                seed_input.text,
            )
        )
        self.add(start_button)
        self.with_background(color=UI_BACKGROUND_COLOUR).with_padding(all=UI_PADDING)


class GameOptionsView(BaseView):
    """Manages the rendering of game option elements on the screen."""

    __slots__ = ()

    def _setup_layout(self: GameOptionsView) -> None:
        """Set up the layout for the view."""
        self.ui.add(self.window.background_image)
        layout = UIBoxLayout(vertical=True, space_between=UI_PADDING)
        layout.add(OptionsPanel())
        layout.add(BackButton())
        self.ui.add(UIAnchorLayout(children=(layout,)))

    def __init__(self: GameOptionsView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the game.
        """
        super().__init__(window)
