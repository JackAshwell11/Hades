"""Stores constants relating to the game and its functionality."""
from __future__ import annotations

# Builtin
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Final

__all__ = (
    "ARMOUR_REGEN_AMOUNT",
    "DAMPING",
    "ENEMY_GENERATE_INTERVAL",
    "ENEMY_GENERATION_DISTANCE",
    "ENEMY_RETRY_COUNT",
    "FOOTPRINT_INTERVAL",
    "FOOTPRINT_LIMIT",
    "GAME_LOGGER",
    "GameObjectType",
    "LOGGING_DICT_CONFIG",
    "MAX_SEE_AHEAD",
    "MAX_VELOCITY",
    "MELEE_RESOLUTION",
    "MOVEMENT_FORCE",
    "OBSTACLE_AVOIDANCE_ANGLE",
    "PATH_POINT_RADIUS",
    "SLOWING_RADIUS",
    "SPRITE_SCALE",
    "SPRITE_SIZE",
    "TARGET_DISTANCE",
    "TOTAL_ENEMY_COUNT",
    "WANDER_CIRCLE_DISTANCE",
    "WANDER_CIRCLE_RADIUS",
)


class GameObjectType(Enum):
    """Stores the different types of game objects that can exist in the game."""

    FLOOR = auto()
    WALL = auto()
    PLAYER = auto()
    ENEMY = auto()
    POTION = auto()


# Create the log directory making sure it exists. Then create the path for the current
# log file
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Logging constants
GAME_LOGGER: Final = "hades"
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

# Sprite sizes
SPRITE_SCALE: Final = 0.5
SPRITE_SIZE: Final = 128 * SPRITE_SCALE

# Physics constants
DAMPING: Final = 0.0001
MAX_VELOCITY: Final = 200

# General game object constants
ARMOUR_REGEN_AMOUNT: Final = 1
FOOTPRINT_INTERVAL: Final = 0.5
FOOTPRINT_LIMIT: Final = 10
MELEE_RESOLUTION: Final = 10
MOVEMENT_FORCE: Final = 100
TARGET_DISTANCE: Final = 3 * SPRITE_SIZE

# Steering constants
MAX_SEE_AHEAD: Final = 2 * SPRITE_SIZE
OBSTACLE_AVOIDANCE_ANGLE: Final = 60
PATH_POINT_RADIUS: Final = 1 * SPRITE_SIZE
SLOWING_RADIUS: Final = 3 * SPRITE_SIZE
WANDER_CIRCLE_DISTANCE: Final = 50
WANDER_CIRCLE_RADIUS: Final = 25

# Enemy generation constants
ENEMY_GENERATE_INTERVAL: Final = 1
ENEMY_GENERATION_DISTANCE: Final = 5
ENEMY_RETRY_COUNT: Final = 3
TOTAL_ENEMY_COUNT: Final = 1
