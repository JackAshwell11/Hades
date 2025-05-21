"""Acts as the entry point to the game by creating and initialising the window."""

from __future__ import annotations

# Builtin
from datetime import UTC, datetime
from logging import getLogger
from logging.config import dictConfig
from pathlib import Path
from typing import Final

# Pip
from arcade import Texture, View, Window, get_default_texture, get_image
from arcade.gui import UIWidget
from PIL.ImageFilter import GaussianBlur

# Custom
from hades import ViewType
from hades.game import Game
from hades.player import Player
from hades.views.start_menu import StartMenu

__all__ = ("HadesWindow",)

# Constants
GAME_LOGGER: Final[str] = "hades"

# The Gaussian blur filter to apply to the background image
BACKGROUND_BLUR: Final[GaussianBlur] = GaussianBlur(5)

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
            "detailed": {
                "format": (
                    "%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(funcName)s:%("
                    "lineno)d - %(message)s"
                ),
            },
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": (
                    log_dir / f"{datetime.now(tz=UTC).strftime('%Y-%m-%d')}.log"
                ),
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5,
                "encoding": "utf8",
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "": {
                "handlers": ["file", "console"],
                "level": "WARNING",
            },
            GAME_LOGGER: {
                "handlers": ["file", "console"],
                "level": "DEBUG",
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
        background_image: The background image of the window.
    """

    def __init__(self: HadesWindow) -> None:
        """Initialise the object."""
        super().__init__()
        self.views: dict[ViewType, View] = {}
        self.background_image: UIWidget = UIWidget(
            width=self.width,
            height=self.height,
        ).with_background(texture=get_default_texture())

    def save_background(self: HadesWindow) -> None:
        """Save the current background image to a texture."""
        self.background_image.with_background(
            texture=Texture(get_image().filter(BACKGROUND_BLUR)),
        )

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
    logger.debug("Initialised window")

    # Initialise the views
    window.views[ViewType.START_MENU] = StartMenu()
    window.views[ViewType.GAME] = Game()
    window.views[ViewType.PLAYER] = Player()
    window.show_view(window.views[ViewType.START_MENU])
    logger.debug("Initialised views")

    # Run the game
    logger.debug("Running game")
    window.run()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()
