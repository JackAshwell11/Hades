"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
from pymunk import Vec2d

# Custom
from hades.game_objects.movements import SteeringObject

if TYPE_CHECKING:
    from hades.game_objects.base import (
        ComponentData,
        ComponentType,
        GameObjectComponent,
    )

__all__ = ("ECS", "ECSError")


class ECSError(Exception):
    """Raised when an error occurs with the ECS."""

    def __init__(
        self: ECSError,
        *,
        not_registered_type: str,
        value: int | str | ComponentType,
        error: str = "not registered with the ECS",
    ) -> None:
        """Initialise the object.

        Args:
            not_registered_type: The game object or component type that is not
                registered.
            value: The value that is not registered.
            error: The problem raised by the ECS.
        """
        super().__init__(
            f"The {not_registered_type} `{value}` is {error}.",
        )


class ECS:
    """Stores and manages game objects registered with the entity component system."""

    __slots__ = (
        "_next_game_object_id",
        "_components",
        "_steering_objects",
    )

    def __init__(self: ECS) -> None:
        """Initialise the object."""
        self._next_game_object_id = 0
        self._components: dict[int, dict[ComponentType, GameObjectComponent]] = {}
        self._steering_objects: dict[int, SteeringObject] = {}

    def add_game_object(
        self: ECS,
        component_data: ComponentData,
        *components: type[GameObjectComponent],
        steering: bool,
    ) -> int:
        """Add a game object to the system with optional components.

        Args:
            component_data: The data for the components.
            *components: The optional list of components for the game object.
            steering: Whether the game object should have a steering object or not.

        Returns:
            The game object ID.

        Raises:
            ECSError: The component type `type` is already registered with the ECS.
        """
        # Create the game object and a steering object if required
        self._components[self._next_game_object_id] = {}
        if steering:
            self._steering_objects[self._next_game_object_id] = SteeringObject(
                self._next_game_object_id,
                Vec2d(0, 0),
                Vec2d(0, 0),
            )

        # Add the optional components to the system
        for component in components:
            if component.component_type in self._components[self._next_game_object_id]:
                del self._components[self._next_game_object_id]
                del self._steering_objects[self._next_game_object_id]
                raise ECSError(
                    not_registered_type="component type",
                    value=component.component_type,
                    error="already registered with the ECS",
                )

            # Initialise the component and add it to the system
            self._components[self._next_game_object_id][component.component_type] = (
                component(self._next_game_object_id, self, component_data)
            )

        # Increment _next_game_object_id and return the current game object ID
        self._next_game_object_id += 1
        return self._next_game_object_id - 1

    def remove_game_object(self: ECS, game_object_id: int) -> None:
        """Remove a game object from the system.

        Args:
            game_object_id: The game object ID.

        Raises:
            ECSError: The game object ID `ID` is not registered with the ECS.
        """
        # Check if the game object is registered or not
        if game_object_id not in self._components:
            raise ECSError(
                not_registered_type="game object ID",
                value=game_object_id,
            )

        # Delete the game object from the system
        del self._components[game_object_id]
        del self._steering_objects[game_object_id]

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
            ECSError: The game object ID `ID` is not registered with the ECS.
        """
        # Check if the game object ID is registered or not
        if game_object_id not in self._components:
            raise ECSError(
                not_registered_type="game object ID",
                value=game_object_id,
            )

        # Return the game object's components
        return self._components[game_object_id]

    def get_component_for_game_object(
        self: ECS,
        game_object_id: int,
        component_type: ComponentType,
    ) -> GameObjectComponent:
        """Get a component for a given game object ID.

        Args:
            game_object_id: The game object ID.
            component_type: The component type to get.

        Returns:
            The component for the given game object ID.

        Raises:
            ECSError: The game object ID `ID` is not registered with the ECS.
            KeyError: The component type is not part of the game object.
        """
        # Check if the game object ID is registered or not
        if game_object_id not in self._components:
            raise ECSError(
                not_registered_type="game object ID",
                value=game_object_id,
            )

        # Return the specified component
        return self._components[game_object_id][component_type]

    def get_steering_object_for_game_object(
        self: ECS,
        game_object_id: int,
    ) -> SteeringObject:
        """Get a steering object for a given game object ID.

        Args:
            game_object_id: The game object ID.

        Returns:
            The steering object for the given game object ID.

        Raises:
            ECSError: The game object ID `ID` is not registered with the ECS.
        """
        # Check if the game object ID is registered or not
        if game_object_id not in self._steering_objects:
            raise ECSError(
                not_registered_type="game object ID",
                value=game_object_id,
            )

        # Return the specified steering object
        return self._steering_objects[game_object_id]

    def get_components_for_component_type(
        self: ECS,
        component_type: ComponentType,
    ) -> list[GameObjectComponent]:
        """Get a list of components for a given component type.

        Args:
            component_type: The component type.

        Returns:
            The list of components for the given component type.
        """
        return [
            components[component_type]
            for components in self._components.values()
            if component_type in components
        ]

    def __repr__(self: ECS) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<EntityComponentSystem (Game object count={len(self._components)})>"
