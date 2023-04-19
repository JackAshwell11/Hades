"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.exceptions import AlreadyAddedComponentError

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentType, GameObjectComponent

__all__ = ("EntityComponentSystem",)


class EntityComponentSystem:
    """Stores and manages the different game objects registered with the system.

    Attributes
    ----------
    game_objects: dict[int, dict[ComponentType, GameObjectComponent]]
        The game objects registered with the system along with their components.
    ids: dict[int, str]
        A mapping of game object ID to the game object's name.
    """

    __slots__ = (
        "_next_game_object_id",
        "game_objects",
        "ids",
    )

    def __init__(self: EntityComponentSystem) -> None:
        """Initialise the object."""
        self._next_game_object_id = 0
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
            self.add_component_to_game_object(self._next_game_object_id, component)

        # Increment next_game_object_id and return the current game object ID
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

    def add_component_to_game_object(
        self: EntityComponentSystem,
        game_object_id: int,
        component: GameObjectComponent,
    ) -> None:
        """Add a component to a game object.

        Parameters
        ----------
        game_object_id: int
            The ID of the game object.
        component: GameObjectComponent
            The instantiated game object component subclasses to add to the game object.

        Raises
        ------
        AlreadyAddedComponentError
            Component already added.
        KeyError
            The game object does not exist in the system.
        """
        # Check if the component type is already added to the game object
        if component.component_type in self.game_objects[game_object_id]:
            raise AlreadyAddedComponentError

        # Add the component to the game object
        self.game_objects[game_object_id][component.component_type] = component
        component.system = self

    def remove_component_from_game_object(
        self: EntityComponentSystem,
        game_object_id: int,
        component_type: ComponentType,
    ) -> None:
        """Remove a component from the game object.

        Parameters
        ----------
        game_object_id: int
            The ID of the game object.
        component_type: ComponentType
            The component type to remove from the game object.

        Raises
        ------
        KeyError
            The game object does not exist in the system or the component is not added
            to the game object.
        """
        del self.game_objects[game_object_id][component_type]

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
