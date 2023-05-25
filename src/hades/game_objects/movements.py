"""Manages the different movement algorithms available to the game objects."""
from __future__ import annotations

# Builtin
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

# Custom
from hades.constants import MOVEMENT_FORCE, ComponentType
from hades.game_objects.base import GameObjectComponent

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentData
    from hades.game_objects.system import ECS

__all__ = ("KeyboardMovement",)


class MovementBase(metaclass=ABCMeta):
    """The base class for all movement algorithms."""

    @abstractmethod
    def calculate_force(self: MovementBase) -> tuple[float, float]:
        """Calculate the new force to apply to the game object.

        Returns:
            The new force to apply to the game object.
        """


class KeyboardMovement(MovementBase, GameObjectComponent):
    """Allows a game object's movement to be controlled by the keyboard."""

    __slots__ = (
        "up_pressed",
        "down_pressed",
        "left_pressed",
        "right_pressed",
    )

    # Class variables
    component_type: ComponentType = ComponentType.KEYBOARD_MOVEMENT

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
        self.up_pressed: bool = False
        self.down_pressed: bool = False
        self.left_pressed: bool = False
        self.right_pressed: bool = False

    def calculate_force(self: KeyboardMovement) -> tuple[float, float]:
        """Calculate the new force to apply to the game object.

        Returns:
            The new force to apply to the game object.
        """
        return (
            MOVEMENT_FORCE * (self.right_pressed - self.left_pressed),
            MOVEMENT_FORCE * (self.up_pressed - self.down_pressed),
        )

    def __repr__(self: KeyboardMovement) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<KeyboardMovement (Up pressed={self.up_pressed}) (Down"
            f" pressed={self.down_pressed}) (Left pressed={self.left_pressed}) (Right"
            f" pressed={self.right_pressed})>"
        )
