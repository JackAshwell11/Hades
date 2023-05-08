"""Stores all the constructors used to make the game objects."""
from __future__ import annotations

# Custom

__all__ = ()


from hades.constants import GameObjectType
from hades.game_objects.components import Inventory
from hades.game_objects.movements import KeyboardMovement
from hades.game_objects.system import ECS
from hades.textures import TextureType

print(
    ECS().add_game_object(
        GameObjectType.PLAYER,
        (5, 2),
        {
            "inventory_width": 5,
            "inventory_height": 7,
            "texture_types": {TextureType.PLAYER_IDLE},
            "blocking": False,
        },
        Inventory,
        KeyboardMovement,
    ),
)
