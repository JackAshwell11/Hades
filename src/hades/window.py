"""Acts as the entry point to the game by creating and initialising the window."""

from __future__ import annotations

# Builtin
import logging.config

# Pip
import arcade

# Custom
from hades.constants import GAME_LOGGER, LOGGING_DICT_CONFIG
from hades.views.start_menu import StartMenu

__all__ = ("Window",)

# Initialise logging and get the game logger
logging.config.dictConfig(LOGGING_DICT_CONFIG)
logger = logging.getLogger(GAME_LOGGER)


class Window(arcade.Window):
    """Manages the window and allows switching between views.

    Attributes:
        views: Holds all the views used by the game.
    """

    def __init__(self: Window) -> None:
        """Initialise the object."""
        super().__init__()
        self.views: dict[str, arcade.View] = {}

    def __repr__(self: Window) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Window (Width={self.width}) (Height={self.height})>"


def main() -> None:
    """Initialise the game and runs it."""
    # Initialise the window
    window = Window()
    window.center_window()

    # Initialise and load the start menu view
    new_view = StartMenu()
    window.views["StartMenu"] = new_view
    window.show_view(window.views["StartMenu"])
    new_view.ui_manager.enable()
    logger.info("Initialised start menu view")

    # Run the game
    logger.info("Running game")
    window.run()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()


# TODO: Start thinking about how to refactor the game (maybe move to pygam or some other
#  library or other methods). Could move events to c++ and/or move physics system to
#  box2d or chipmunk among other stuff
