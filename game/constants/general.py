"""Stores general constants that can't really be grouped together."""
from __future__ import annotations

# Builtin
import pathlib
from datetime import datetime

# Pip
import arcade

__all__ = (
    "CONSUMABLE_LEVEL_MAX_RANGE",
    "DAMPING",
    "DEBUG_ATTACK_DISTANCE",
    "DEBUG_GAME",
    "DEBUG_LINES",
    "DEBUG_VECTOR_FIELD_LINE",
    "DEBUG_VIEW_DISTANCE",
    "ENEMY_LEVEL_MAX_RANGE",
    "GAME_LOGGER",
    "INVENTORY_HEIGHT",
    "INVENTORY_WIDTH",
    "LEVEL_GENERATOR_INTERVAL",
    "LOGGING_DICT_CONFIG",
)

# Create the log directory making sure it exists. Then create the path for the current
# log file
log_dir = pathlib.Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Logging constants
GAME_LOGGER = "game"
LOGGING_DICT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": (
                "[%(asctime)s %(levelname)s] [%(filename)s:%(funcName)s():%(lineno)d] -"
                " %(message)s"
            ),
        }
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
DEBUG_LINES = False
DEBUG_GAME = True
DEBUG_VIEW_DISTANCE = arcade.color.RED
DEBUG_ATTACK_DISTANCE = arcade.color.BLUE
DEBUG_VECTOR_FIELD_LINE = arcade.color.YELLOW

# Physics constants
DAMPING = 0

# Inventory constants
INVENTORY_WIDTH = 6
INVENTORY_HEIGHT = 5

# Enemy and consumable level generator constants
LEVEL_GENERATOR_INTERVAL = 10
ENEMY_LEVEL_MAX_RANGE = 5
CONSUMABLE_LEVEL_MAX_RANGE = 5
