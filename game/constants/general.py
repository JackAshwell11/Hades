from __future__ import annotations

# Pip
import arcade

# Custom
from game.constants.generation import TileType

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

# Consumable constants
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
HEALTH_BOOST_POTION_INCREASE = 50
HEALTH_BOOST_POTION_DURATION = 10
ARMOUR_BOOST_POTION_INCREASE = 10
ARMOUR_BOOST_POTION_DURATION = 10
SPEED_BOOST_POTION_INCREASE = 200
SPEED_BOOST_POTION_DURATION = 5
FIRE_RATE_BOOST_POTION_INCREASE = -0.5
FIRE_RATE_BOOST_POTION_DURATION = 5
