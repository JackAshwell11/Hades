"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentType, GameObjectComponent

__all__ = ("ECS", "NotRegisteredError")


class NotRegisteredError(Exception):
    """Raised when a game object or component type is not registered."""

    def __init__(
        self: NotRegisteredError,
        *,
        not_registered_type: str,
        value: int | ComponentType,
    ) -> None:
        """Initialise the object.

        Args:
            not_registered_type: The game object or component type that is not
                registered.
            value: The value that is not registered.
        """
        super().__init__(
            f"The {not_registered_type} `{value}` is not registered with the ECS.",
        )


class ECS:
    """Stores and manages game objects registered with the entity component system."""

    __slots__ = (
        "_next_game_object_id",
        "_game_objects",
    )

    def __init__(self: ECS) -> None:
        """Initialise the object."""
        self._next_game_object_id = 0
        self._game_objects: defaultdict[
            int,
            dict[ComponentType, GameObjectComponent],
        ] = defaultdict(dict)

    def add_game_object(self: ECS, *components: GameObjectComponent) -> int:
        """Add a game object to the system with optional components.

        Args:
            *components: A list of instantiated game object component subclasses which
                belong to the game object.

        Returns:
            The ID of the created game object.

        Raises:
            NotRegisteredError: The component type `type` is not registered with the
                ECS.
        """
        # Create the game object
        self._game_objects[self._next_game_object_id] = {}

        # Add the optional components to the system
        for component in components:
            # Check its dependencies are registered in the system
            for dependency in component.dependencies:
                if dependency not in self._game_objects[self._next_game_object_id]:
                    del self._game_objects[self._next_game_object_id]
                    raise NotRegisteredError(
                        not_registered_type="component type",
                        value=dependency,
                    )

            # Add the component to the system
            self._game_objects[self._next_game_object_id][
                component.component_type
            ] = component
            component.system = self

        # Increment _next_game_object_id and return the current game object ID
        self._next_game_object_id += 1
        return self._next_game_object_id - 1

    def remove_game_object(self: ECS, game_object_id: int) -> None:
        """Remove a game object from the system.

        Args:
            game_object_id: The game object ID.

        Raises:
            NotRegisteredError: The game object `ID` is not registered with the ECS.
        """
        # Check if the game object is registered or not
        if game_object_id not in self._game_objects:
            raise NotRegisteredError(
                not_registered_type="game object",
                value=game_object_id,
            )

        # Delete the game object from the system
        del self._game_objects[game_object_id]

    def get_components_for_game_object(
        self: ECS,
        game_object_id: int,
    ) -> dict[ComponentType, GameObjectComponent]:
        """Get a game object's components.

        Args:
            game_object_id: The game object ID.

        Returns:
            The game object's components.

        Raises:
            NotRegisteredError: The game object `ID` is not registered with the ECS.
        """
        # Check if the game object ID is registered or not
        if game_object_id not in self._game_objects:
            raise NotRegisteredError(
                not_registered_type="game object",
                value=game_object_id,
            )

        # Return the game object's components
        return self._game_objects[game_object_id]

    def __repr__(self: ECS) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<EntityComponentSystem (Game object count={len(self._game_objects)})>"
