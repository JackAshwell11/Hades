from __future__ import annotations

# Pip
import arcade

# Debug constants
DEBUG_LINES = False
DEBUG_GAME = True
DEBUG_VIEW_DISTANCE = arcade.color.RED
DEBUG_ATTACK_DISTANCE = arcade.color.BLUE
DEBUG_VECTOR_FIELD_LINE = arcade.color.YELLOW
LOGGING_FORMAT = (
    "[%(asctime)s %(levelname)s] [%(filename)s:%(funcName)s():%(lineno)d] - %(message)s"
)

# Physics constants
DAMPING = 0

# Inventory constants
INVENTORY_WIDTH = 6
INVENTORY_HEIGHT = 5

# Enemy and consumable level generator constants
LEVEL_GENERATOR_INTERVAL = 10
ENEMY_LEVEL_MAX_RANGE = 5
CONSUMABLE_LEVEL_MAX_RANGE = 5
