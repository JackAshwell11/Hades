"""Manages the different movement algorithms available to the game objects."""
from __future__ import annotations

# Builtin
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, cast

# Custom
from hades.game_objects.attributes import MovementForce
from hades.game_objects.base import ComponentType, GameObjectComponent

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentData
    from hades.game_objects.system import ECS

__all__ = ("KeyboardMovement", "MovementBase", "SteeringMovement")


class MovementBase(GameObjectComponent, metaclass=ABCMeta):
    """The base class for all movement algorithms."""

    __slots__ = ("movement_force",)

    # Class variables
    component_type: ComponentType = ComponentType.MOVEMENTS

    def __init__(
        self: MovementBase,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.movement_force: MovementForce = cast(
            MovementForce,
            self.system.get_component_for_game_object(
                self.game_object_id,
                ComponentType.MOVEMENT_FORCE,
            ),
        )

    @abstractmethod
    def calculate_force(self: MovementBase) -> tuple[float, float]:
        """Calculate the new force to apply to the game object.

        Returns:
            The new force to apply to the game object.
        """


class KeyboardMovement(MovementBase, GameObjectComponent):
    """Allows a game object's movement to be controlled by the keyboard."""

    __slots__ = (
        "north_pressed",
        "south_pressed",
        "east_pressed",
        "west_pressed",
    )

    def __init__(
        self: KeyboardMovement,
        game_object_id: int,
        system: ECS,
        _: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
        """
        super().__init__(game_object_id, system, _)
        self.north_pressed: bool = False
        self.south_pressed: bool = False
        self.east_pressed: bool = False
        self.west_pressed: bool = False

    def calculate_force(self: KeyboardMovement) -> tuple[float, float]:
        """Calculate the new force to apply to the game object.

        Returns:
            The new force to apply to the game object.
        """
        return (
            self.movement_force.value * (self.east_pressed - self.west_pressed),
            self.movement_force.value * (self.north_pressed - self.south_pressed),
        )

    def __repr__(self: KeyboardMovement) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<KeyboardMovement (North pressed={self.north_pressed}) (South"
            f" pressed={self.south_pressed}) (East pressed={self.east_pressed}) (West"
            f" pressed={self.west_pressed})>"
        )


class SteeringMovement(MovementBase, GameObjectComponent):
    """Allows a game object's movement to be controlled by steering algorithms."""

    __slots__ = ()

    def __init__(
        self: SteeringMovement,
        game_object_id: int,
        system: ECS,
        _: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
        """
        super().__init__(game_object_id, system, _)

    def calculate_force(self: SteeringMovement) -> tuple[float, float]:
        """Calculate the new force to apply to the game object.

        Returns:
            The new force to apply to the game object.
        """
        return 0, 0

    def __repr__(self: SteeringMovement) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return "<SteeringMovement>"
