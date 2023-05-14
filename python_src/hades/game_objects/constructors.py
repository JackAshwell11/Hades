"""Stores all the constructors used to make the game objects."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, NamedTuple

# Custom
from hades.constants import GameObjectType
from hades.game_objects.components import Inventory
from hades.game_objects.movements import KeyboardMovement
from hades.textures import TextureType

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentData, GameObjectComponent

__all__ = ("ENEMY", "FLOOR", "PLAYER", "POTION", "WALL", "GameObjectConstructor")


class GameObjectConstructor(NamedTuple):
    """Represents a constructor for a game object."""

    game_object_type: GameObjectType
    texture_types: set[TextureType]
    component_data: ComponentData = {}
    components: list[type[GameObjectComponent]] = []
    blocking: bool = False


# Static tiles
WALL = GameObjectConstructor(GameObjectType.WALL, {TextureType.WALL}, blocking=True)
FLOOR = GameObjectConstructor(GameObjectType.FLOOR, {TextureType.FLOOR})

# Player characters
PLAYER = GameObjectConstructor(
    GameObjectType.PLAYER,
    {TextureType.PLAYER_IDLE},
    component_data={"inventory": (6, 5)},
    components=[Inventory, KeyboardMovement],
)

# Enemy characters
ENEMY = GameObjectConstructor(
    GameObjectType.ENEMY,
    {TextureType.ENEMY_IDLE},
)

# Potion tiles
POTION = GameObjectConstructor(
    GameObjectType.POTION,
    {TextureType.HEALTH_POTION},
)
