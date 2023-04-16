"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.game_objects.ecs.enums import ComponentType

if TYPE_CHECKING:
    from hades.game_objects.ecs.enums import GameObjectComponent

__all__ = ("System",)


# Define
GameObject = dict[ComponentType, str]


class System:
    """Stores and manages the different game objects registered with the system."""

    __slots__ = ("_next_game_object_id", "_components", "_game_objects")

    def __init__(self: System) -> None:
        """Initialise the object."""
        self._next_game_object_id = 0
        self._components: dict[ComponentType, set[int]] = {}
        self._game_objects: dict[
            int,
            dict[ComponentType, type[GameObjectComponent]],
        ] = {}

    def add_game_object(self: System) -> None:
        pass

    # def delete_game_object


# TODO: So system will have collection of game objects which are represented with a dict
#  with key being component enum and value being instantiated component. Arcade.sprite
#  should be a graphics component and will be added to a spritelist on initialisation.
#  Game objects can be put into groups inside system (entities, tiles, particles). USE
#  https://github.com/avikor/entity_component_system/tree/master/ecs AND
#  https://github.com/benmoran56/esper/blob/master/esper/__init__.py (MAINLY THIS ONE)

# TODO: DETERMINE HOW TO STORE COMPONENTS AND GAME OBJECTS. SHOULD PROCESSORS BE USED?
#  SHOULD GAMEOBJECTCOMPONENT BE USED? SHOULD _COMPONENTS AND _GAME_OBJECTS BE USED?
