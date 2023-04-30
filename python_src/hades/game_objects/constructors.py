"""Stores all the constructors used to make the game objects."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from hades_extensions import TileType

if TYPE_CHECKING:
    from hades.game_objects.base import GameObjectComponent

__all__ = ("CONSUMABLES", "ENEMY", "FLOOR", "PLAYER", "WALL")

# Player game objects
PLAYER: tuple[GameObjectComponent, ...] = ()

# Enemy game objects
ENEMY: tuple[GameObjectComponent, ...] = ()

# Tile game objects
FLOOR: tuple[GameObjectComponent, ...] = ()
WALL: tuple[GameObjectComponent, ...] = ()

# Consumable game objects
CONSUMABLES: dict[TileType, tuple[GameObjectComponent, ...]] = {
    TileType.HealthPotion: (),
    TileType.ArmourPotion: (),
    TileType.HealthBoostPotion: (),
    TileType.ArmourBoostPotion: (),
    TileType.SpeedBoostPotion: (),
    TileType.FireRateBoostPotion: (),
}
