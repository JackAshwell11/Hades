from __future__ import annotations

# Pip
import arcade

# Custom
from constants.enums import TileType

# Debug constants
DEBUG_LINES = False
DEBUG_GAME = True
DEBUG_VIEW_DISTANCE = arcade.color.RED
DEBUG_ATTACK_DISTANCE = arcade.color.BLUE
LOGGING_FORMAT = (
    "[%(asctime)s  %(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s()] -"
    " %(message)s"
)

# Sprite sizes
SPRITE_SCALE = 2.5
SPRITE_SIZE = 16 * SPRITE_SCALE

# Physics constants
DAMPING = 0

# Inventory constants
INVENTORY_WIDTH = 6
INVENTORY_HEIGHT = 5

# Item constants
ENEMIES = [TileType.ENEMY]
CONSUMABLES = [TileType.HEALTH_POTION, TileType.HEALTH_BOOST_POTION]
HEALTH_POTION_INCREASE = 10
