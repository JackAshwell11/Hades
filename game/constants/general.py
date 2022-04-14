from __future__ import annotations

# Pip
import arcade

# Custom
from constants.generation import TileType

# Debug constants
DEBUG_LINES = False
DEBUG_GAME = True
DEBUG_VIEW_DISTANCE = arcade.color.RED
DEBUG_ATTACK_DISTANCE = arcade.color.BLUE
LOGGING_FORMAT = (
    "[%(asctime)s %(levelname)s] [%(filename)s:%(funcName)s():%(lineno)d] - %(message)s"
)

# Physics constants
DAMPING = 0

# Inventory constants
INVENTORY_WIDTH = 6
INVENTORY_HEIGHT = 5

# Item constants
ENEMIES = [TileType.ENEMY]
CONSUMABLES = [
    TileType.HEALTH_POTION,
    TileType.ARMOUR_POTION,
    TileType.HEALTH_BOOST_POTION,
    TileType.ARMOUR_BOOST_POTION,
    TileType.SPEED_BOOST_POTION,
    TileType.FIRE_RATE_BOOST_POTION,
]
HEALTH_POTION_INCREASE = 20
ARMOUR_POTION_INCREASE = 10
