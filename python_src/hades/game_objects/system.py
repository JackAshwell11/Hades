"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.exceptions import AlreadyAddedComponentError

if TYPE_CHECKING:
    from hades.game_objects.base import (
        ComponentType,
        GameObjectComponent,
        ProcessorComponent,
    )

__all__ = ("EntityComponentSystem",)


class EntityComponentSystem:
    """Stores and manages the different game objects registered with the system.

    Attributes
    ----------
    game_objects: dict[int, dict[ComponentType, type[GameObjectComponent]]]
        The game objects registered with the system along with their components.
    processors: list[type[ProcessorComponent]]
        The processors registered with the system.
    """

    __slots__ = (
        "_next_game_object_id",
        "game_objects",
        "processors",
    )

    def __init__(self: EntityComponentSystem) -> None:
        """Initialise the object."""
        self._next_game_object_id = 0
        self.game_objects: dict[
            int,
            dict[ComponentType, type[GameObjectComponent]],
        ] = {}
        self.processors: list[type[ProcessorComponent]] = []

    def add_game_object(
        self: EntityComponentSystem,
        *components: type[GameObjectComponent],
    ) -> int:
        """Add a game object to the system with optional components.

        Returns
        -------
        int
            The ID of the created game object.
        """
        # Create a new game object with its optional components
        for component in components:
            # Test if the game object exists. If not, create it
            if self._next_game_object_id not in self.game_objects:
                self.game_objects[self._next_game_object_id] = {}

            # Add the component to the game object
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

    def add_component_to_game_object(
        self: EntityComponentSystem,
        game_object_id: int,
        component: type[GameObjectComponent],
    ) -> None:
        """Add a component to a game object.

        Parameters
        ----------
        game_object_id: int
            The ID of the game object.
        component: type[GameObjectComponent]
            The instantiated game object component subclasses to add to the game object.

        Raises
        ------
        AlreadyAddedComponentError
            Component already added.
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

    def add_processor(
        self: EntityComponentSystem,
        processor: type[ProcessorComponent],
    ) -> None:
        """Add a processor to the system.

        Parameters
        ----------
        processor: type[ProcessorComponent]
            The instantiated processor component subclasses to add to the system.
        """
        # Add the processor to the system
        self.processors.append(processor)
        processor.system = self

    def remove_processor(
        self: EntityComponentSystem,
        processor_type: ComponentType,
    ) -> None:
        """Remove a processor from the system.

        Parameters
        ----------
        processor_type: ComponentType
            The processor type to remove from the system.

        Raises
        ------
        ValueError
            The processor does not exist in the system.
        """
        # Remove all instances of processors that have a type of processor_type
        for processor in self.processors:
            if processor.component_type is processor_type:
                self.processors.remove(processor)

    def __repr__(self: EntityComponentSystem) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<EntityComponentSystem (Game object count={len(self.game_objects)})>"
