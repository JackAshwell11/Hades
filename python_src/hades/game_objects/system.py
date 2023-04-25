"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from hades.game_objects.base import ComponentType, GameObjectComponent

__all__ = ("ECS", "NotRegisteredError")

# Define a generic type for the keyword arguments
KW = TypeVar("KW")
R = TypeVar("R")


class NotRegisteredError(Exception):
    """Raised when a game object/component type/event is not registered."""

    def __init__(
        self: NotRegisteredError,
        *,
        not_registered_type: str,
        value: int | str | ComponentType,
    ) -> None:
        """Initialise the object.

        Parameters
        ----------
        not_registered_type: str
            The game object/component type/event that is not registered.
        value: int | str | ComponentType
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
        "_components",
        "_event_handlers",
    )

    def __init__(self: ECS) -> None:
        """Initialise the object."""
        self._next_game_object_id = 0
        self._game_objects: defaultdict[int, set[GameObjectComponent]] = defaultdict(
            set,
        )
        self._components: defaultdict[ComponentType, set[int]] = defaultdict(set)
        self._event_handlers: defaultdict[str, set[Callable[[KW], R]]] = defaultdict(
            set,
        )

    @staticmethod
    def _get_event_names(component: GameObjectComponent) -> Generator[str, None, None]:
        """Get a component's events.

        Parameters
        ----------
        component: GameObjectComponent
            The component to get events for.

        Returns
        -------
        Generator[str, str, None]
            A generator of the component's events.
        """
        return (i for i in dir(component) if i.startswith("event_"))

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
        self._game_objects[self._next_game_object_id] = set()

        # Add the optional components and their events to the system
        for component in components:
            self._components[component.component_type].add(self._next_game_object_id)
            self._game_objects[self._next_game_object_id].add(component)
            component.system = self
            for event_name in self._get_event_names(component):
                self._event_handlers[event_name].add(getattr(component, event_name))

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

        # Remove all events relating to this game object
        for components in self._game_objects[game_object_id]:
            for name in self._get_event_names(components):
                self._event_handlers[name].remove(getattr(components, name))

        # Delete the game object from the system
        del self._game_objects[game_object_id]
        for ids in self._components.values():
            ids.remove(game_object_id)

    def dispatch_event(self: ECS, event: str, **kwargs: KW) -> None:
        """Dispatch an event with keyword arguments.

        Parameters
        ----------
        event: str
            The event name.
        """
        # Check if the event is registered or not
        if event not in self._event_handlers:
            raise NotRegisteredError(not_registered_type="event", value=event)

        # Dispatch the event to all handlers
        for handler in self._event_handlers.get(event, []):
            handler(**kwargs)

    def get_components_for_game_object(
        self: ECS,
        game_object_id: int,
    ) -> set[GameObjectComponent]:
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
        set[GameObjectComponent]
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

    def get_game_objects_for_component_type(
        self: ECS,
        component_type: ComponentType,
    ) -> set[int]:
        """Get a component type's game objects.

        Parameters
        ----------
        component_type: ComponentType
            The component type.

        Raises
        ------
        NotRegisteredError
            The game object is not registered.

        Returns
        -------
        set[int]
            The component type's game objects.
        """
        # Check if the component type is registered or not
        if component_type not in self._components:
            raise NotRegisteredError(
                not_registered_type="component type",
                value=component_type.name,
            )

        # Return the component type's game objects
        return self._components[component_type]

    def get_handlers_for_event_name(self: ECS, event: str) -> set[Callable[[KW], R]]:
        """Get an event's handlers.

        Parameters
        ----------
        event: str
            The event name.

        Raises
        ------
        NotRegisteredError
            The event is not registered.

        Returns
        -------
        set[Callable[[KW], R]]
            The event's handlers.
        """
        # Check if the event is registered or not
        if event not in self._event_handlers:
            raise NotRegisteredError(not_registered_type="event", value=event)

        # Return the event's handlers
        return self._event_handlers[event]

    def __repr__(self: ECS) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return (
            "<EntityComponentSystem (Game object"
            f" count={len(self._game_objects)}) (Event"
            f" count={len(self._event_handlers)})>"
        )


# TODO: Dispatch_event could possibly return results (need to see if good idea) by
#  returning collection containing result for each handler (could sort by component name
#  to remove Nones). It could also become a global function and global private variable
