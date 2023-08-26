"""Contains the functionality which manages the game objects."""
from __future__ import annotations

# Builtin
from typing import Final

# Custom
from hades.game_objects.systems.attacks import AttackSystem
from hades.game_objects.systems.attributes import (
    ArmourRegenSystem,
    GameObjectAttributeSystem,
)
from hades.game_objects.systems.inventory import InventorySystem
from hades.game_objects.systems.movements import (
    FootprintSystem,
    KeyboardMovementSystem,
    SteeringMovementSystem,
)

___all__ = (
    "ArmourRegenSystem",
    "AttackSystem",
    "FootprintSystem",
    "GameObjectAttributeError",
    "GameObjectAttributeSystem",
    "InventorySpaceError",
    "InventorySystem",
    "KeyboardMovementSystem",
    "SYSTEMS",
    "SteeringMovementSystem",
)


# Create a set of systems that exist in the game
SYSTEMS: Final = {
    AttackSystem,
    GameObjectAttributeSystem,
    ArmourRegenSystem,
    InventorySystem,
    SteeringMovementSystem,
    KeyboardMovementSystem,
    FootprintSystem,
}
