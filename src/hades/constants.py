"""Stores constants relating to the game and its functionality."""
from __future__ import annotations

# Builtin
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

# Pip
from arcade import color

__all__ = (
    "ARMOUR_REGEN_AMOUNT",
    "DAMPING",
    "DEBUG_ENEMY_SPAWN_COLOR",
    "DEBUG_ENEMY_SPAWN_SIZE",
    "DEBUG_GAME",
    "ENEMY_GENERATE_INTERVAL",
    "ENEMY_GENERATION_DISTANCE",
    "ENEMY_RETRY_COUNT",
    "GAME_LOGGER",
    "GameObjectType",
    "LOGGING_DICT_CONFIG",
    "MAX_VELOCITY",
    "MELEE_RESOLUTION",
    "MOVEMENT_FORCE",
    "SPRITE_SCALE",
    "SPRITE_SIZE",
    "TOTAL_ENEMY_COUNT",
)


# The different types of game objects in the game
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
GAME_LOGGER = "hades"
LOGGING_DICT_CONFIG = {
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
            "filename": log_dir.joinpath(f"{datetime.now().strftime('%Y-%m-%d')}.log"),
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

# Debug constants
DEBUG_GAME = True
DEBUG_ENEMY_SPAWN_COLOR = color.RED
DEBUG_ENEMY_SPAWN_SIZE = 5

# Sprite sizes
SPRITE_SCALE = 0.4375
SPRITE_SIZE = 128 * SPRITE_SCALE

# Physics constants
DAMPING = 0.000001
MAX_VELOCITY = 50

# General game object constants
MOVEMENT_FORCE = 100
ARMOUR_REGEN_AMOUNT = 1
MELEE_RESOLUTION = 10

# Enemy generation constants
TOTAL_ENEMY_COUNT = 8
ENEMY_RETRY_COUNT = 3
ENEMY_GENERATE_INTERVAL = 1
ENEMY_GENERATION_DISTANCE = 5
