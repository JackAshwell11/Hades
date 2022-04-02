from __future__ import annotations

# Builtin
import logging
import pathlib

# Pip
import arcade

# Custom
from constants.general import DO_LOGGING, LOGGING_FORMAT
from views.start_menu import StartMenu

# Create the log path
log_path = (
    pathlib.Path(__file__).resolve().parent.joinpath("saves").joinpath("game.log")
)


class Window(arcade.Window):
    """
    Manages the window and allows switching between views.

     Attributes
    ----------
    views: dict[str, arcade.View]
        Holds all the views used by the game.
    logger: logging.Logger
        The logger used for logging information to a file.
    """

    def __init__(self) -> None:
        super().__init__()
        self.views: dict[str, arcade.View] = {}
        self.logger: logging.Logger = logging.getLogger("game")

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
    if DO_LOGGING:
        arcade_logger = logging.getLogger("arcade")
        arcade_logger.setLevel(logging.INFO)
        game_logger = logging.getLogger("game")
        game_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(LOGGING_FORMAT)
        file_handler.setFormatter(formatter)
        arcade_logger.addHandler(file_handler)
        game_logger.addHandler(file_handler)

    # Run the game
    window.run()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()
