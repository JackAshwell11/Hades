"""Acts as the entry point to the game by creating and initialising the window."""

from __future__ import annotations

# Builtin
from datetime import UTC, datetime
from logging import getLogger
from logging.config import dictConfig
from pathlib import Path
from typing import Final

# Pip
from arcade import View, Window

# Custom
from hades.views.start_menu import StartMenu

__all__ = ("HadesWindow",)

# Constants
GAME_LOGGER: Final[str] = "hades"

# Create the log directory making sure it exists. Then create the path for the current
# log file
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Initialise logging and get the game logger
dictConfig(
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
                "filename": (
                    log_dir / f"{datetime.now(tz=UTC).strftime('%Y-%m-%d')}.log"
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
logger = getLogger(GAME_LOGGER)


class HadesWindow(Window):
    """Manages the window and allows switching between views.

    Attributes:
        views: Holds all the views used by the game.
    """

    def __init__(self: HadesWindow) -> None:
        """Initialise the object."""
        super().__init__()
        self.views: dict[str, View] = {}

    def __repr__(self: HadesWindow) -> str:  # pragma: no cover
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Window (Width={self.width}) (Height={self.height})>"


def main() -> None:
    """Initialise the game and runs it."""
    # Initialise the window
    window = HadesWindow()
    window.center_window()

    # Initialise and load the start menu view
    new_view = StartMenu()
    window.views["StartMenu"] = new_view
    window.show_view(new_view)
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
