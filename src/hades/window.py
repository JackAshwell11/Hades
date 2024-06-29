"""Acts as the entry point to the game by creating and initialising the window."""

from __future__ import annotations

# Builtin
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

# Pip
import arcade

# Custom
from hades.views.start_menu import StartMenu

__all__ = ("Window",)

# Constants
GAME_LOGGER: Final[str] = "hades"

# Create the log directory making sure it exists. Then create the path for the current
# log file
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Initialise logging and get the game logger
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": (
                    "[%(asctime)s %(levelname)s] [%(filename)s:%(funcName)s():%(lineno)"
                    "d] - %(message)s"
                ),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "default",
                "filename": log_dir.joinpath(
                    f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}.log",
                ),
                "maxBytes": 5242880,  # 5MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            GAME_LOGGER: {
                "level": "DEBUG",
                "handlers": ["file"],
                "propagate": False,
            },
        },
    },
)
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

    def __repr__(self: Window) -> str:  # pragma: no cover
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


# TODO: Maybe move to pygame
# TODO: Maybe move events to C++
# TODO: Maybe move physics system to Box2D
# TODO: Maybe add sanitizers to C++
