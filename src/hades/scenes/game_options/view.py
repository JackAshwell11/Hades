"""Manages the rendering of game option elements on the screen."""

from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from arcade import get_window
from arcade.gui import UIAnchorLayout, UIBoxLayout, UIFlatButton

# Custom
from hades import MENU_WIDGET_SPACING, UI_PADDING, BackButton
from hades.scenes.base.view import BaseView
from hades_engine import DifficultyLevel

if TYPE_CHECKING:
    from arcade.gui.style import StyleRef

    from hades.window import HadesWindow

__all__ = ("GameOptionsView",)


class StartButton(UIFlatButton):
    """Represents a button to start the game."""

    __slots__ = ()

    def __init__(self: StartButton) -> None:
        """Initialise the object."""
        super().__init__(text="Start Game")
        self.on_click = (  # type: ignore[method-assign]
            lambda _: get_window().dispatch_event("on_start_level")  # type: ignore[assignment]
        )


class DifficultyButton(UIFlatButton):
    """Represents a button to select a difficulty level.

    Attributes:
        clicked: Whether the button has been clicked or not.
    """

    __slots__ = ()

    def __init__(self: DifficultyButton, level: DifficultyLevel) -> None:
        """Initialise the object.

        Args:
            level: The difficulty level for the button.
        """
        self.clicked: bool = False
        super().__init__(text=level.name)
        self.on_click = (  # type: ignore[method-assign]
            lambda _: get_window().dispatch_event(  # type: ignore[assignment]
                "on_difficulty_change",
                level,
            )
        )

    def get_current_style(self: DifficultyButton) -> StyleRef | None:
        """Get the current style of the button.

        Returns:
            The current style of the button.
        """
        if self.clicked:
            return self.style.get("press")
        return super().get_current_style()  # type: ignore[no-any-return]


class DifficultyLayout(UIBoxLayout):
    """Represents a layout for selecting the difficulty level."""

    __slots__ = ("buttons",)

    def __init__(self: DifficultyLayout) -> None:
        """Initialise the object."""
        super().__init__(space_between=UI_PADDING)
        self.buttons: dict[DifficultyLevel, DifficultyButton] = {
            level: DifficultyButton(level)
            for level in (
                DifficultyLevel.Easy,
                DifficultyLevel.Normal,
                DifficultyLevel.Hard,
            )
        }
        self.add(
            UIBoxLayout(
                vertical=False,
                children=self.buttons.values(),
                space_between=UI_PADDING,
            ),
        )


class GameOptionsView(BaseView):
    """Manages the rendering of game option elements on the screen."""

    __slots__ = ("difficulty_layout",)

    def _setup_layout(self: GameOptionsView) -> None:
        """Set up the layout for the view."""
        vertical_box = UIBoxLayout(space_between=MENU_WIDGET_SPACING)
        vertical_box.add(StartButton())
        vertical_box.add(self.difficulty_layout)
        vertical_box.add(BackButton())
        self.difficulty_layout.buttons[
            self.window.model.game_state.difficulty_level
        ].clicked = True
        self.ui.add(UIAnchorLayout(children=(vertical_box,)))

    def __init__(self: GameOptionsView, window: HadesWindow) -> None:
        """Initialise the object.

        Args:
            window: The window for the game.
        """
        self.difficulty_layout: DifficultyLayout = DifficultyLayout()
        super().__init__(window)
