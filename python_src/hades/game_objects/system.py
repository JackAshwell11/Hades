"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from collections import defaultdict
from typing import TYPE_CHECKING

# Custom
from hades.game_objects.base import ComponentType
from hades.game_objects.components import HadesSprite

if TYPE_CHECKING:
    from hades.constants import GameObjectType
    from hades.game_objects.base import ComponentData, GameObjectComponent

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
        "_components",
        "_ids",
    )

    def __init__(self: ECS) -> None:
        """Initialise the object."""
        self._next_game_object_id = 0
        self._components: dict[int, dict[ComponentType, GameObjectComponent]] = {}
        self._ids: defaultdict[GameObjectType, set[int]] = defaultdict(set)

    def add_game_object(
        self: ECS,
        game_object_type: GameObjectType,
        position: tuple[int, int],
        component_data: ComponentData,
        *components: type[GameObjectComponent],
    ) -> HadesSprite:
        """Add a game object to the system with optional components.

        Args:
            game_object_type: The type of the game object.
            position: The position of the game object on the screen.
            component_data: The data for the components.
            *components: The optional list of components for the game object.

        Returns:
            The instantiated sprite object.

        Raises:
            NotRegisteredError: The component type `type` is not registered with the
                ECS.
        """
        # Create the game object and get the constructor for this game object type
        sprite_obj = HadesSprite(game_object_type, position, component_data)
        self._components[self._next_game_object_id] = {ComponentType.SPRITE: sprite_obj}
        self._ids[game_object_type].add(self._next_game_object_id)

        # Add the optional components to the system
        for component in components:
            self._components[self._next_game_object_id][component.component_type] = (
                component(component_data)
            )
            component.system = self

        # Increment _next_game_object_id and return the result
        self._next_game_object_id += 1
        return sprite_obj

    def remove_game_object(self: ECS, game_object_id: int) -> None:
        """Remove a game object from the system.

        Args:
            game_object_id: The game object ID.

        Raises:
            NotRegisteredError: The game object/sprite object for the game object ID
            `ID` is not registered with the ECS.
        """
        # Check if the game object is registered or not
        if game_object_id not in self._components:
            raise NotRegisteredError(
                not_registered_type="game object",
                value=game_object_id,
            )

        # Delete the game object from the system
        sprite_obj = self._components[game_object_id][ComponentType.SPRITE]
        if isinstance(sprite_obj, HadesSprite):
            sprite_obj.remove_from_sprite_lists()
            del self._components[game_object_id]
            self._ids[sprite_obj.game_object_type].remove(game_object_id)
        else:
            raise NotRegisteredError(
                not_registered_type="sprite object for the game object ID",
                value=game_object_id,
            )

    def get_component_for_game_object(
        self: ECS,
        game_object_id: int,
        component_type: ComponentType,
    ) -> GameObjectComponent:
        """Get a component from a game object.

        Args:
            game_object_id: The game object ID.
            component_type: The component type to get.

        Returns:
            The game object's component.

        Raises:
            NotRegisteredError: The game object `ID` is not registered with the ECS.
            KeyError: The component type is not part of the game object.
        """
        # Check if the game object ID is registered or not
        if game_object_id not in self._components:
            raise NotRegisteredError(
                not_registered_type="game object",
                value=game_object_id,
            )

        # Return the game object's components
        return self._components[game_object_id][component_type]

    def __repr__(self: ECS) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<EntityComponentSystem (Game object count={len(self._components)})>"
