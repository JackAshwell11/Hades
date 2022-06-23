"""Stores code that is shared between all views simplifying development."""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade
from arcade.gui import (
    UIAnchorWidget,
    UILayout,
    UIManager,
    UIMouseFilterMixin,
    UITextArea,
)

if TYPE_CHECKING:
    from game.views.game_view import Game
    from game.window import Window

__all__ = ("BaseView",)

# Get the logger
logger = logging.getLogger(__name__)


class DisappearingInfoBox(UIMouseFilterMixin, UIAnchorWidget):
    """Represents a simple dialog box that pops up with a message and disappears after a
    certain amount of time.

    Parameters
    ----------
    parent_view: BaseView
        The parent view which created this box.
    width: float
        The width of the message box.
    height: float
        The height of the message box.
    message_text: str
        The text to display.
    text_color: int
        The color of the text in the box.
    background_color: arcade.Color
        The color of the background of the box.
    disappear_time: float
        The time before the box should disappear.
    """

    def __init__(
        self,
        parent_view: BaseView,
        *,
        width: float = 400,
        height: float = 150,
        message_text: str,
        text_color: arcade.Color = arcade.color.BLACK,
        background_color: arcade.Color = arcade.color.BABY_BLUE,
        disappear_time: float = 3,
    ) -> None:
        # The offset used for the anchoring
        anchor_offset = 10

        # Store various variables needed for this box to function
        self._parent_view: BaseView = parent_view
        self._time_counter: float = disappear_time

        # Set up the text box
        self._text_area = UITextArea(
            width=width - anchor_offset,
            height=height - anchor_offset,
            text=message_text,
            font_size=18,
            text_color=text_color,
        )

        # Set up the layout
        group = UILayout(
            width=width,
            height=height,
            children=[
                UIAnchorWidget(
                    child=self._text_area,
                    anchor_x="left",
                    anchor_y="top",
                    align_x=10,
                    align_y=-10,
                ),
            ],
        ).with_background(
            arcade.Texture.create_filled(
                "background color", (int(width), int(height)), background_color
            )
        )
        logger.info(
            "Created info box with text `%s` and time %f", message_text, disappear_time
        )

        super().__init__(child=group, anchor_y="bottom", align_y=anchor_offset)

    def __repr__(self) -> str:
        return f"<DisappearingInfoBox (Text={self._text_area.text})>"

    def on_update(self, delta_time: float) -> None:
        """Updates the internal time counter and checks to see if the box should
        disappear.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the counter
        self._time_counter -= delta_time

        # Check if the box should disappear
        if self._time_counter <= 0:
            self.remove_box()

    def remove_box(self) -> None:
        """Removes the box from the UI manager."""
        self.parent.remove(self)
        self._parent_view.current_info_box = None
        logger.info("Info box has disappeared with text %s", self._text_area.text)


class BackButton(arcade.gui.UIFlatButton):
    """A button which will switch back to the game view."""

    def __repr__(self) -> str:
        return (
            f"<BackButton (Position=({self.center_x}, {self.center_y}))"
            f" (Width={self.width}) (Height={self.height})>"
        )

    def on_click(self, _) -> None:
        """Called when the button is clicked."""
        # Get the current window and view
        window: Window = arcade.get_window()

        # Show the game view
        game_view: Game = window.views["Game"]  # noqa
        window.show_view(game_view)


class BaseView(arcade.View):
    """The base class for all views.

    Attributes
    ----------
    ui_manager: UIManager
        Manages all the different UI elements for this view.
    background_color: arcade.Color
        The background color of this view.
    current_info_box: DisappearingInfoBox | None
        The currently displayed info box.
    """

    def __init__(self) -> None:
        super().__init__()
        self.window: Window = self.window
        self.ui_manager: UIManager = UIManager()
        self.background_color: arcade.Color = arcade.color.BABY_BLUE
        self.current_info_box: DisappearingInfoBox | None = None

    def __repr__(self) -> str:
        return f"<BaseView (Current window={self.window})>"

    def on_show_view(self) -> None:
        """Called when the view is shown."""
        self.window.background_color = self.background_color
        self.ui_manager.enable()
        self.post_show_view()
        logger.info("Shown view %r", self)

    def on_hide_view(self) -> None:
        """Called when the view is hidden."""
        self.ui_manager.disable()
        self.post_hide_view()
        logger.info("Hid view %r", self)

    def post_show_view(self) -> None:
        """Called after the view is shown allowing for extra functionality to be
        added."""

    def post_hide_view(self) -> None:
        """Called after the view is hidden allowing for extra functionality to be
        added."""

    def display_info_box(self, text: str) -> None:
        """Displays an info box that disappears after a set amount of time.

        Parameters
        ----------
        text: str
            The text to display to the user.
        """
        if self.current_info_box is None:
            self.current_info_box = DisappearingInfoBox(self, message_text=text)
            self.ui_manager.add(self.current_info_box)

    @staticmethod
    def add_back_button(vertical_box: arcade.gui.UIBoxLayout) -> None:
        """Adds the back button to a given vertical box.

        Parameters
        ----------
        vertical_box: arcade.gui.UIBoxLayout
            The UIBoxLayout to add the back button too.
        """
        vertical_box.add(BackButton(text="Back", width=200).with_space_around(top=20))
