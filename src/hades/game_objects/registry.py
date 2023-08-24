"""Manages the registry and all of its components and systems."""
from __future__ import annotations

# Builtin
from copy import deepcopy
from typing import TYPE_CHECKING, TypeVar, cast, overload

# Custom
from hades.game_objects.base import ComponentBase, SystemBase
from hades.game_objects.steering import KinematicObject, Vec2d

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ("Registry", "RegistryError")


# Define some type variables for the registry
C = TypeVar("C", bound=ComponentBase)
COne = TypeVar("COne", bound=ComponentBase)
CTwo = TypeVar("CTwo", bound=ComponentBase)
S = TypeVar("S", bound=SystemBase)


class RegistryError(Exception):
    """Raised when an error occurs with the registry."""

    def __init__(
        self: RegistryError,
        *,
        not_registered_type: str,
        value: int | str | type[S],
        error: str = "is not registered with the registry",
    ) -> None:
        """Initialise the object.

        Args:
            not_registered_type: The type of item that is not registered.
            value: The value that is not registered.
            error: The problem raised by the registry.
        """
        super().__init__(f"The {not_registered_type} `{value}` {error}.")


class Registry:
    """Manages game objects, components, and systems that are registered."""

    __slots__ = (
        "_next_game_object_id",
        "_components",
        "_game_objects",
        "_systems",
        "_kinematic_objects",
        "walls",
    )

    def __init__(self: Registry) -> None:
        """Initialise the object."""
        self._next_game_object_id = 0
        self._components: dict[type[ComponentBase], set[int]] = {}
        self._game_objects: dict[int, dict[type[ComponentBase], ComponentBase]] = {}
        self._systems: dict[type[SystemBase], SystemBase] = {}
        self._kinematic_objects: dict[int, KinematicObject] = {}
        self.walls: set[Vec2d] = set()

    def update(self: Registry, delta_time: float) -> None:
        """Update all the systems.

        Args:
            delta_time: The time interval since the last time the function was called.
        """
        for system in self._systems.values():
            system.update(delta_time)

    def create_game_object(
        self: Registry,
        *components: ComponentBase,
        kinematic: bool = False,
    ) -> int:
        """Create a game object.

        Args:
            components: The components to add to the game object.
            kinematic: Whether the game object should have a kinematic object or not.

        Returns:
            The game object ID.
        """
        # Add the game object to the system
        self._game_objects[self._next_game_object_id] = {}
        if kinematic:
            self._kinematic_objects[self._next_game_object_id] = KinematicObject(
                Vec2d(0, 0),
                Vec2d(0, 0),
            )

        # Add the game object to the components
        for component in components:
            self._components.setdefault(type(component), set()).add(
                self._next_game_object_id,
            )
            self._game_objects.setdefault(self._next_game_object_id, {}).setdefault(
                type(component),
                deepcopy(component),
            )

        # Increment the game object ID and return the current game object ID
        self._next_game_object_id += 1
        return self._next_game_object_id - 1

    def delete_game_object(self: Registry, game_object_id: int) -> None:
        """Delete a game object.

        Args:
            game_object_id: The game object ID.

        Raises:
            RegistryError: The game object ID `ID` is not registered with the registry.
        """
        # Check if the game object is registered or not
        if game_object_id not in self._game_objects:
            raise RegistryError(
                not_registered_type="game object ID",
                value=game_object_id,
            )

        # Delete the game object from the system
        del self._game_objects[game_object_id]
        for game_objects in self._components.values():
            if game_object_id in game_objects:
                game_objects.remove(game_object_id)
        if game_object_id in self._kinematic_objects:
            del self._kinematic_objects[game_object_id]

    def add_system(self: Registry, system: SystemBase) -> None:
        """Add a system to the registry.

        Args:
            system: The system to add.
        """
        # Check if the system is already registered or not
        if type(system) in self._systems:
            raise RegistryError(
                not_registered_type="system",
                value=system.__class__.__name__,
                error="is already registered with the registry",
            )

        # Add the system to the registry
        self._systems[type(system)] = system

    def get_system(self: Registry, system: type[S]) -> S:
        """Get a system from the registry.

        Args:
            system: The system to get.

        Raises:
            RegistryError: The system `system` is not registered with the registry.

        Returns:
            The system.
        """
        # Check if the system is registered or not
        if system not in self._systems:
            raise RegistryError(
                not_registered_type="system",
                value=system,
            )

        # Return the specified system
        return cast(S, self._systems[system])

    def get_component_for_game_object(
        self: Registry,
        game_object_id: int,
        component: type[C],
    ) -> C:
        """Get a component from the registry for a game object.

        Args:
            game_object_id: The game object ID.
            component: The component to get.

        Raises:
            RegistryError: The game object ID `ID` is not registered with the registry.
            KeyError: The component type is not part of the game object.

        Returns:
            The component from the registry.
        """
        # Check if the game object ID is registered or not
        if game_object_id not in self._game_objects:
            raise RegistryError(
                not_registered_type="game object ID",
                value=game_object_id,
            )

        # Return the specified component
        return cast(C, self._game_objects[game_object_id][component])

    @overload
    def get_components(
        self: Registry,
        __component: type[C],
    ) -> Generator[tuple[int, tuple[C]], None, None]:  # pragma: no cover
        ...

    @overload
    def get_components(
        self: Registry,
        __component: type[C],
        __component_two: type[COne],
    ) -> Generator[tuple[int, tuple[C, COne]], None, None]:  # pragma: no cover
        ...

    @overload
    def get_components(
        self: Registry,
        __component: type[C],
        __component_two: type[COne],
        __component_three: type[CTwo],
    ) -> Generator[tuple[int, tuple[C, COne, CTwo]], None, None]:  # pragma: no cover
        ...

    def get_components(
        self: Registry,
        *components: type[ComponentBase],
    ) -> Generator[tuple[int, ComponentBase | tuple[ComponentBase, ...]], None, None]:
        """Get the components for all game objects that have the required components.

        Args:
            *components: The components to check for.

        Returns:
            The game object IDs and their components.
        """
        # Get the game object IDs that have the specified components. If no game objects
        # are registered, the result should be an empty set
        game_object_ids: set[int] = set.intersection(
            *(self._components.get(component, set()) for component in components),
        )

        # Return the game object IDs and their components
        return (
            (
                game_object_id,
                tuple(
                    self._game_objects[game_object_id][component]
                    for component in components
                ),
            )
            for game_object_id in game_object_ids
        )

    def get_kinematic_object_for_game_object(
        self: Registry,
        game_object_id: int,
    ) -> KinematicObject:
        """Get a kinematic object for a given game object ID.

        Args:
            game_object_id: The game object ID.

        Returns:
            The kinematic object for the given game object ID.

        Raises:
            RegistryError: The game object ID `ID` does not have a kinematic object.
        """
        # Check if the game object ID is registered or not
        if game_object_id not in self._kinematic_objects:
            raise RegistryError(
                not_registered_type="game object ID",
                value=game_object_id,
                error="does not have a kinematic object",
            )

        # Return the specified kinematic object
        return self._kinematic_objects[game_object_id]

    def __repr__(self: Registry) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<Registry (Game object count={len(self._game_objects)}) (Component"
            f" count={len(self._components)}) (System count={len(self._systems)})>"
        )
