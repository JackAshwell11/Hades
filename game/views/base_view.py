from __future__ import annotations

# Builtin
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Pip
import arcade
from arcade.gui import (
    UIAnchorWidget,
    UIEvent,
    UILayout,
    UIManager,
    UIMouseFilterMixin,
    UITextArea,
)

if TYPE_CHECKING:
    from game.window import Window

# Get the logger
logger = logging.getLogger(__name__)


@dataclass
class UIBoxDisappearEvent(UIEvent):
    """Dispatched when an info box disappears."""


class DisappearingInfoBox(UIMouseFilterMixin, UIAnchorWidget):
    """
    Represents a simple dialog box that pops up with a message and disappears after a
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
            f"Created info box with text `{message_text}` and time {disappear_time}"
        )

        super().__init__(child=group, anchor_y="bottom", align_y=anchor_offset)

    def on_update(self, delta_time: float) -> None:
        """
        Updates the internal time counter and checks to see if the box should disappear.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the counter
        self._time_counter -= delta_time

        # Check if the box should disappear
        if self._time_counter <= 0:
            self.remove_box(UIBoxDisappearEvent(self))

    def remove_box(self, _) -> None:
        """Removes the box from the UI manager."""
        self.parent.remove(self)
        self._parent_view.current_info_box = None


class BaseView(arcade.View):
    """
    The base class for all views.

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
        logger.info(f"Shown view ({self.__repr__()})")

    def on_hide_view(self) -> None:
        """Called when the view is hidden."""
        self.ui_manager.disable()
        self.post_hide_view()
        logger.info(f"Hid view ({self.__repr__()})")

    def post_show_view(self) -> None:
        """Called after the view is shown allowing for extra functionality to be
        added."""
        return None

    def post_hide_view(self) -> None:
        """Called after the view is hidden allowing for extra functionality to be
        added."""
        return None

    def display_info_box(self, text: str) -> None:
        """
        Displays an info box that disappears after a set amount of time.

        Parameters
        ----------
        text: str
            The text to display to the user.
        """
        if self.current_info_box is None:
            self.current_info_box = DisappearingInfoBox(self, message_text=text)
            self.ui_manager.add(self.current_info_box)
