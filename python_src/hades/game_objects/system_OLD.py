"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from collections import deque
from typing import TYPE_CHECKING

# Custom

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentType, GameObjectComponent

__all__ = ("EntityComponentSystem",)


class EntityComponentSystem:
    """Stores and manages the different game objects registered with the system.

    Attributes
    ----------
    game_objects: listdict[ComponentType, GameObjectComponent]]
        The game objects registered with the system along with their components.
    ids: dict[int, str]
        A mapping of game object ID to the game object's name.
    """

    __slots__ = (
        "_available_game_objects",
        "game_objects",
        "ids",
    )

    def __init__(self: EntityComponentSystem) -> None:
        """Initialise the object."""
        self._available_game_objects: deque[int] = deque()
        self.game_objects: dict[int, dict[ComponentType, GameObjectComponent]] = {}
        self.ids: dict[int, str] = {}

    def add_game_object(
        self: EntityComponentSystem,
        name: str,
        *components: GameObjectComponent,
    ) -> int:
        """Add a game object to the system with optional components.

        Parameters
        ----------
        name: str
            The name of the game object.

        Returns
        -------
        int
            The ID of the created game object.
        """
        # Create the game object and add the optional components to it
        self.game_objects[self._next_game_object_id] = {}
        self.ids[self._next_game_object_id] = name
        for component in components:
            self.game_objects[self._next_game_object_id][
                component.component_type
            ] = component
            component.system = self

        # Increment _next_game_object_id and return the current game object ID
        self._next_game_object_id += 1
        return self._next_game_object_id - 1

    def remove_game_object(self: EntityComponentSystem, game_object_id: int) -> None:
        """Remove a game object from the system.

        Parameters
        ----------
        game_object_id: int
            The ID of the game object.

        Raises
        ------
        KeyError
            The game object does not exist in the system.
        """
        del self.game_objects[game_object_id]
        del self.ids[game_object_id]

    def __repr__(self: EntityComponentSystem) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<EntityComponentSystem (Game object count={len(self.game_objects)})>"


# TODO: Game objects can maybe be put into groups inside system (entities, tiles,
#  particles). USE https://github.com/avikor/entity_component_system/tree/master/ecs AND
#  https://github.com/benmoran56/esper/blob/master/esper/__init__.py (MAINLY THIS ONE)

# TODO: DETERMINE HOW TO STORE COMPONENTS AND GAME OBJECTS. SHOULD PROCESSORS BE USED?
#  SHOULD GAMEOBJECTCOMPONENT BE USED? SHOULD A COMPONENTS AND GAME_OBJECTS DICT BE
#  USED?
