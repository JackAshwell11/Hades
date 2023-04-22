"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.base import ComponentType, GameObjectComponent

__all__ = ("ECS",)


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
        self._event_handlers: defaultdict[str, set[Callable]] = defaultdict(set)

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

        # Add the optional components to the system
        for component in components:
            self._components[component.component_type].add(self._next_game_object_id)
            self._game_objects[self._next_game_object_id].add(component)
            component.system = self
            for event_name in (i for i in component.__dir__() if i.startswith("on_")):
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
        # Remove all events relating to this game object
        for component in self._game_objects[game_object_id]:
            for name, handler in component.events:
                self._event_handlers[name].remove(handler)

        # Delete the game object from the system
        del self._game_objects[game_object_id]
        for component in self._components.values():
            component.remove(game_object_id)

    def dispatch_event(self: ECS, event: str, **kwargs) -> None:
        """Dispatch an event with keyword arguments.

        Parameters
        ----------
        event: str
            The event name.
        """
        for handler in self._event_handlers.get(event, []):
            handler(**kwargs)

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
