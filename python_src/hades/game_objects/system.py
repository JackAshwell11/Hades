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

        Parameters
        ----------
        not_registered_type: str
            The game object or component type that is not registered.
        value: int | ComponentType
            The value that is not registered.
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

        Parameters
        ----------
        *components: type[GameObjectComponent]
            A list of instantiated game object component subclasses which belong to the
            game object.

        Returns
        -------
        int
            The ID of the created game object.
        """
        # Create the game object
        self._game_objects[self._next_game_object_id] = {}

        # Add the optional components to the system
        for component in components:
            self._game_objects[self._next_game_object_id][
                component.component_type
            ] = component
            component.system = self

        # Increment _next_game_object_id and return the current game object ID
        self._next_game_object_id += 1
        return self._next_game_object_id - 1

    def remove_game_object(self: ECS, game_object_id: int) -> None:
        """Remove a game object from the system.

        Parameters
        ----------
        game_object_id: int
            The game object ID.

        Raises
        ------
        KeyError
            The game object does not exist in the system.
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

        Parameters
        ----------
        game_object_id: int
            The game object ID.

        Raises
        ------
        NotRegisteredError
            The game object is not registered.

        Returns
        -------
        dict[ComponentType, GameObjectComponent]
            The game object's components.
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

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<EntityComponentSystem (Game object count={len(self._game_objects)})>"


# TODO: Could probably store IDs in game view as dict with game object type as key and
#  value being set of ints

# TODO: See if more advanced typing could be used anywhere in game_objects/
