"""Stores all the constructors used to make the game objects."""
from __future__ import annotations

# Pip
from hades_extensions import TileType

__all__ = ("CONSUMABLES", "ENEMY", "FLOOR", "PLAYER", "WALL")

# Player game objects
PLAYER = ()

# Enemy game objects
ENEMY = ()

# Tile game objects
FLOOR = ()
WALL = ()

# Consumable game objects
CONSUMABLES = {
    TileType.HealthPotion: (),
    TileType.ArmourPotion: (),
    TileType.HealthBoostPotion: (),
    TileType.ArmourBoostPotion: (),
    TileType.SpeedBoostPotion: (),
    TileType.FireRateBoostPotion: (),
}
