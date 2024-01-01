"""Stores constants relating to the game and its functionality."""

from __future__ import annotations

# Builtin
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Final

# Custom
from hades_extensions.game_objects import SPRITE_SIZE

__all__ = (
    "ATTACK_COOLDOWN",
    "ATTACK_RANGE",
    "BULLET_VELOCITY",
    "COLLECTIBLE_TYPES",
    "DAMAGE",
    "DAMPING",
    "ENEMY_GENERATE_INTERVAL",
    "ENEMY_GENERATION_DISTANCE",
    "ENEMY_RETRY_COUNT",
    "GAME_LOGGER",
    "GameObjectType",
    "INDICATOR_BAR_BORDER_SIZE",
    "INDICATOR_BAR_DISTANCE",
    "INDICATOR_BAR_HEIGHT",
    "INDICATOR_BAR_WIDTH",
    "LOGGING_DICT_CONFIG",
    "MAX_BULLET_RANGE",
    "MAX_VELOCITY",
    "TOTAL_ENEMY_COUNT",
    "USABLE_TYPES",
)


class GameObjectType(Enum):
    """Stores the different types of game objects that can exist in the game."""

    ENEMY = auto()
    FLOOR = auto()
    PLAYER = auto()
    POTION = auto()
    WALL = auto()


# Create the log directory making sure it exists. Then create the path for the current
# log file
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Logging constants
GAME_LOGGER: Final[str] = "hades"
LOGGING_DICT_CONFIG: Final = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": (
                "[%(asctime)s %(levelname)s] [%(filename)s:%(funcName)s():%(lineno)d] -"
                " %(message)s"
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
}

# Attack constants
ATTACK_COOLDOWN: Final = 3
ATTACK_RANGE: Final = 3 * SPRITE_SIZE
BULLET_VELOCITY: Final = 300
DAMAGE: Final = 10
MAX_BULLET_RANGE: Final = 10 * SPRITE_SIZE

# Enemy generation constants
ENEMY_GENERATE_INTERVAL: Final[int] = 1
ENEMY_GENERATION_DISTANCE: Final[int] = 5
ENEMY_RETRY_COUNT: Final[int] = 3
TOTAL_ENEMY_COUNT: Final[int] = 1

# Indicator bar constants
INDICATOR_BAR_BORDER_SIZE: Final[int] = 4
INDICATOR_BAR_DISTANCE: Final[int] = 32
INDICATOR_BAR_HEIGHT: Final[int] = 10
INDICATOR_BAR_WIDTH: Final[int] = 50

# Physics constants
DAMPING: Final[float] = 0.0001
MAX_VELOCITY: Final[int] = 200

# Define some collections for game object types
COLLECTIBLE_TYPES: Final[set[GameObjectType]] = {
    GameObjectType.POTION,
}
USABLE_TYPES: Final[set[GameObjectType]] = {
    GameObjectType.POTION,
}
