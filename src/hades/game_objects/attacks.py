"""Manages the different attack algorithms available."""
from __future__ import annotations

# Builtin
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

# Custom
from hades.game_objects.base import ComponentType, GameObjectComponent

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentData
    from hades.game_objects.system import ECS

__all__ = ("AttackBase", "AttackManager")


class AttackBase(GameObjectComponent, metaclass=ABCMeta):
    """The base class for all attack algorithms."""

    def __init__(
        self: AttackBase,
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

    @abstractmethod
    def perform_attack(self: AttackBase) -> None:
        """Perform an attack on other game objects."""


class AttackManager(GameObjectComponent):
    """Allows a game object to attack and be attacked by other game objects.

    Attributes:
        current_attack: The index of the currently selected attack algorithm.
    """

    __slots__ = ("attacks", "current_attack")

    # Class variables
    component_type: ComponentType = ComponentType.ATTACK_MANAGER

    def __init__(
        self: AttackManager,
        game_object_id: int,
        system: ECS,
        _: ComponentData,
        attacks: list[AttackBase],
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            attacks: A list of attack algorithms that can be performed by a game object.
        """
        super().__init__(game_object_id, system, _)
        self.attacks: list[AttackBase] = attacks
        self.current_attack: int = 0

    def run_algorithm(self: AttackManager) -> None:
        """Run the currently selected attack algorithm."""
        self.attacks[self.current_attack].perform_attack()

    def __repr__(self: AttackManager) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<AttackManager (Attack algorithm count={len(self.attacks)})>"


# TODO: Maybe look at optimising AttackManager. It may be unnecessary
