from __future__ import annotations

# Pip
import arcade

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

# Enemy and consumable normal distribution constants
DISTRIBUTION_GENERATOR_INTERVAL = 10
ENEMY_NORMAL_MAX_RANGE = 3
CONSUMABLE_NORMAL_MAX_RANGE = 3
