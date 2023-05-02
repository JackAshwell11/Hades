"""Stores all the constructors used to make the game objects."""
from __future__ import annotations

# Pip
from hades_extensions import TileType

# Custom
from hades.game_objects.base import GameObjectConstructor
from hades.game_objects.components import Graphics
from hades.textures import TextureType

__all__ = ("CONSUMABLES", "ENEMY", "FLOOR", "PLAYER", "WALL")


# Player game objects
PLAYER = GameObjectConstructor(
    (Graphics,),
    {"texture_types": {TextureType.PLAYER_IDLE}},
)

# Enemy game objects
ENEMY = GameObjectConstructor((Graphics,), {"texture_types": {TextureType.ENEMY_IDLE}})

# Tile game objects
FLOOR = GameObjectConstructor((Graphics,), {"texture_types": {TextureType.FLOOR}})
WALL = GameObjectConstructor((Graphics,), {"texture_types": {TextureType.WALL}})

# Consumable game objects
CONSUMABLES = {
    TileType.HealthPotion: GameObjectConstructor(
        (Graphics,),
        {"texture_types": {TextureType.HEALTH_POTION}},
    ),
    TileType.ArmourPotion: GameObjectConstructor(
        (Graphics,),
        {"texture_types": {TextureType.ARMOUR_POTION}},
    ),
    TileType.HealthBoostPotion: GameObjectConstructor(
        (Graphics,),
        {"texture_types": {TextureType.HEALTH_BOOST_POTION}},
    ),
    TileType.ArmourBoostPotion: GameObjectConstructor(
        (Graphics,),
        {"texture_types": {TextureType.ARMOUR_BOOST_POTION}},
    ),
    TileType.SpeedBoostPotion: GameObjectConstructor(
        (Graphics,),
        {"texture_types": {TextureType.SPEED_BOOST_POTION}},
    ),
    TileType.FireRateBoostPotion: GameObjectConstructor(
        (Graphics,),
        {"texture_types": {TextureType.FIRE_RATE_BOOST_POTION}},
    ),
}
