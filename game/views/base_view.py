from __future__ import annotations

# Builtin
import logging

# Pip
import arcade
import arcade.gui

# Get the logger
logger = logging.getLogger(__name__)


class BaseView(arcade.View):
    """
    The base class for all views.

    Attributes
    ----------
    ui_manager: arcade.gui.UIManager
        Manages all the different UI elements for this view.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ui_manager: arcade.gui.UIManager = arcade.gui.UIManager()
        self.background_color: arcade.Color = arcade.color.BABY_BLUE

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
