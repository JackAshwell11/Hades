"""Acts as the entry point to the game by creating and initialising the window."""

from __future__ import annotations

# Builtin
from datetime import UTC, datetime
from logging.config import dictConfig
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

# Pip
import pygame
from PIL.ImageFilter import GaussianBlur
from pygame import Window
from pygame.time import Clock

# Custom
from hades import ViewType
from hades.views.start_menu import StartMenu

if TYPE_CHECKING:
    from hades.scene import Scene

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


class HadesWindow(Window):
    """Manages the window and allows switching between scenes.

    Attributes:
        scenes: Holds the different scenes used by the game.
    """

    __slots__ = ("clock", "current_scene", "scenes")

    def __init__(self: HadesWindow) -> None:
        """Initialise the object."""
        super().__init__(title="Hades", size=(1280, 720))
        self.scenes: dict[ViewType, Scene] = {}
        self.current_scene: Scene = cast("Scene", None)
        self.clock: Clock = Clock()

    def change_scene(self: HadesWindow, scene_type: ViewType) -> None:
        """Change the current scene to the specified type.

        Args:
            scene_type: The type of the scene to switch to.
        """
        self.current_scene = self.scenes[scene_type]

    def run(self: HadesWindow) -> None:
        """Run the main game loop."""
        running = True
        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False

            self.current_scene.handle_events(events)
            self.current_scene.draw(self.get_surface())
            self.flip()
            self.clock.tick()


def main() -> None:
    """Initialise the game and runs it."""
    # Initialise the window
    window = HadesWindow()

    # Initialise the views
    window.scenes[ViewType.START_MENU] = StartMenu(window)
    # window.scenes[ViewType.GAME] = Game()
    # window.scenes[ViewType.PLAYER] = Player()
    window.change_scene(ViewType.START_MENU)

    # Run the game
    window.run()
    pygame.quit()


# Only make sure the game is run from this file
if __name__ == "__main__":
    main()
