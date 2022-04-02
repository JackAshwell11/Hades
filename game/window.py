from __future__ import annotations

# Builtin
import logging.config

# Pip
import arcade

# Custom
from constants.general import LOGGING_DICT
from views.start_menu import StartMenu

# Get the logger
logger = logging.getLogger(__name__)


class Window(arcade.Window):
    """
    Manages the window and allows switching between views.

     Attributes
    ----------
    views: dict[str, arcade.View]
        Holds all the views used by the game.
    """

    def __init__(self) -> None:
        super().__init__()
        self.views: dict[str, arcade.View] = {}

    def __repr__(self) -> str:
        return f"<Window (Width={self.width}) (Height={self.height})>"


def main() -> None:
    """Initialises the game and runs it."""
    # Initialise the window
    window = Window()
    window.center_window()

    # Initialise and load the start menu view
    new_view = StartMenu()
    window.views["StartMenu"] = new_view
    window.show_view(window.views["StartMenu"])
    new_view.manager.enable()

    # Initialise logging
    logging.config.dictConfig(LOGGING_DICT)

    # Run the game
    window.run()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()
