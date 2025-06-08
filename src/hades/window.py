"""Acts as the entry point to the game by creating and initialising the window."""

from __future__ import annotations

# Builtin
from datetime import UTC, datetime
from logging.config import dictConfig
from pathlib import Path
from typing import TYPE_CHECKING, Final

# Pip
from arcade import Texture, Window, get_default_texture, get_image
from arcade.gui import UIWidget
from arcade.resources import resolve
from PIL.ImageFilter import GaussianBlur

# Custom
from hades import ViewType
from hades.game import Game
from hades.model import HadesModel
from hades.player import Player
from hades.start_menu import StartMenu

if TYPE_CHECKING:
    from hades.view import BaseView

__all__ = ("HadesWindow",)

# Constants
GAME_LOGGER: Final[str] = "hades"

# The Gaussian blur filter to apply to the background image
BACKGROUND_BLUR: Final[GaussianBlur] = GaussianBlur(5)

# The path to the shop offerings JSON file
SHOP_OFFERINGS: Final[Path] = resolve(":resources:shop_offerings.json")

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


class HadesWindow(Window):
    """Manages the window and allows switching between views.

    Attributes:
        views: Holds all the views used by the game.
        background_image: The background image of the window.
        model: The model providing access to the game engine and its functionality.
    """

    __slots__ = ("background_image", "model", "views")

    def __init__(self: HadesWindow) -> None:
        """Initialise the object."""
        super().__init__()
        self.model: HadesModel = HadesModel()
        self.background_image: UIWidget = UIWidget(
            width=self.width,
            height=self.height,
        ).with_background(texture=get_default_texture())
        self.views: dict[ViewType, BaseView] = {
            ViewType.START_MENU: StartMenu(),
            ViewType.GAME: Game(),
            ViewType.PLAYER: Player(),
        }

    def setup(self: HadesWindow) -> None:
        """Set up the window and its views."""
        self.center_window()
        for view in self.views.values():
            view.add_callbacks()
        self.model.game_engine.create_game_objects()
        self.model.game_engine.setup_shop(str(SHOP_OFFERINGS))
        self.show_view(self.views[ViewType.START_MENU])

    def save_background(self: HadesWindow) -> None:
        """Save the current background image to a texture."""
        self.background_image.with_background(
            texture=Texture(get_image().filter(BACKGROUND_BLUR)),
        )


def main() -> None:
    """Initialise the game and run it."""
    window = HadesWindow()
    window.setup()
    window.run()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()
