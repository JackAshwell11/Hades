from __future__ import annotations

# Builtin
import logging
import pathlib
from datetime import datetime

# Pip
import arcade

# Custom
from constants.general import LOGGING_FORMAT
from views.start_menu import StartMenu


def get_log_path() -> pathlib.Path:
    """
    Gets the path to use for the logging output. This takes into account already
    existing files.

    Returns
    -------
    pathlib.Path
        The path to use for the logging output.
    """
    base_path = pathlib.Path(__file__).resolve().parent / "logs"
    base_filename = datetime.now().strftime("%Y-%m-%d")
    file_count = len(list(base_path.glob(f"{base_filename}*.log")))
    return base_path / f"{base_filename}-{file_count+1}.log"


class ArcadeFilter(logging.Filter):
    """A logging filter which removes all arcade debug logs."""

    def __init__(self) -> None:
        super().__init__("arcade")

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filters out arcade debug logs to stop crowding of the log file.

        Parameters
        ----------
        record: logging.LogRecord
            The log record from the arcade logger.

        Returns
        -------
        bool
            Whether to keep the log record or not.
        """
        return record.levelname != "DEBUG"


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
    # Initialise logging
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(get_log_path())
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    handler.addFilter(ArcadeFilter())
    root_logger.addHandler(handler)

    # Initialise the window
    window = Window()
    window.center_window()

    # Initialise and load the start menu view
    new_view = StartMenu()
    window.views["StartMenu"] = new_view
    window.show_view(window.views["StartMenu"])
    new_view.manager.enable()

    # Run the game
    window.run()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()
