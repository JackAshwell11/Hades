"""Stores constants relating to the game and its functionality."""
from __future__ import annotations

# Builtin
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

# Pip
import arcade

__all__ = (
    "ARMOUR_INDICATOR_BAR_COLOR",
    "ARMOUR_REGEN_AMOUNT",
    "ARMOUR_REGEN_WAIT",
    "BULLET_VELOCITY",
    "CONSUMABLE_LEVEL_MAX_RANGE",
    "DAMPING",
    "DEBUG_ATTACK_DISTANCE",
    "DEBUG_ENEMY_SPAWN_COLOR",
    "DEBUG_ENEMY_SPAWN_SIZE",
    "DEBUG_GAME",
    "DEBUG_VECTOR_FIELD_LINE",
    "DEBUG_VIEW_DISTANCE",
    "ENEMY_GENERATE_INTERVAL",
    "ENEMY_INDICATOR_BAR_OFFSET",
    "ENEMY_RETRY_COUNT",
    "FACING_LEFT",
    "FACING_RIGHT",
    "GAME_LOGGER",
    "GameObjectType",
    "HEALTH_INDICATOR_BAR_COLOR",
    "INDICATOR_BAR_BORDER_SIZE",
    "LEVEL_GENERATOR_INTERVAL",
    "LOGGING_DICT_CONFIG",
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
DEBUG_VIEW_DISTANCE = arcade.color.RED
DEBUG_ATTACK_DISTANCE = arcade.color.BLUE
DEBUG_VECTOR_FIELD_LINE = arcade.color.YELLOW
DEBUG_ENEMY_SPAWN_COLOR = arcade.color.RED
DEBUG_ENEMY_SPAWN_SIZE = 5

# Sprite sizes
SPRITE_SCALE = 0.4375
SPRITE_SIZE = 128 * SPRITE_SCALE

# Physics constants
DAMPING = 0

# General game object constants
MOVEMENT_FORCE = 1000000
FACING_RIGHT = 0
FACING_LEFT = 1
ARMOUR_REGEN_WAIT = 5
ARMOUR_REGEN_AMOUNT = 1
BULLET_VELOCITY = 300
MELEE_RESOLUTION = 10
INDICATOR_BAR_BORDER_SIZE = 4
ENEMY_INDICATOR_BAR_OFFSET = 32
HEALTH_INDICATOR_BAR_COLOR = arcade.color.RED
ARMOUR_INDICATOR_BAR_COLOR = arcade.color.SILVER

# Enemy and level generator constants
LEVEL_GENERATOR_INTERVAL = 10
CONSUMABLE_LEVEL_MAX_RANGE = 5
TOTAL_ENEMY_COUNT = 8
ENEMY_RETRY_COUNT = 3
ENEMY_GENERATE_INTERVAL = 1
