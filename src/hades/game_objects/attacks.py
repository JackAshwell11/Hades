"""Manages the different attack algorithms available."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from hades.game_objects.base import AttackAlgorithms, ComponentType, GameObjectComponent

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.base import ComponentData
    from hades.game_objects.system import ECS

__all__ = ("Attacks", "ranged_attack", "melee_attack", "area_of_effect_attack")


def area_of_effect_attack() -> None:
    """Perform an area of effect attack around the game object."""
    raise NotImplementedError


def melee_attack() -> None:
    """Perform a melee attack in the direction the game object is facing."""
    raise NotImplementedError


def ranged_attack() -> None:
    """Perform a ranged attack in the direction the game object is facing."""
    raise NotImplementedError


# Determine the attack algorithm type for each attack algorithm
ATTACKS = {
    AttackAlgorithms.AREA_OF_EFFECT_ATTACK: area_of_effect_attack,
    AttackAlgorithms.MELEE_ATTACK: melee_attack,
    AttackAlgorithms.RANGED_ATTACK: ranged_attack,
}


class Attacks(GameObjectComponent):
    """Allows a game object to attack other game objects."""

    __slots__ = ("_attacks", "_current_attack")

    # Class variables
    component_type: ComponentType = ComponentType.ATTACKS

    def __init__(
        self: Attacks,
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
        self._attacks: list[Callable[[], None]] = [
            ATTACKS[component_type]
            for component_type in component_data["enabled_attacks"]
        ]
        self._current_attack: int = 0

    @property
    def attacks(self: Attacks) -> list[Callable[[], None]]:
        """Get the currently enabled attack algorithms.

        Returns:
            The currently enabled attack algorithms.
        """
        return self._attacks

    @property
    def current_attack(self: Attacks) -> int:
        """Get the index of the current attack algorithm.

        Returns:
            The index of the current attack algorithm.
        """
        return self._current_attack

    def do_attack(self: Attacks) -> None:
        """Perform the currently selected attack algorithm."""
        self._attacks[self._current_attack]()

    def previous_attack(self: Attacks) -> None:
        """Select the previous attack algorithm."""
        self._current_attack = max(self._current_attack - 1, 0)

    def next_attack(self: Attacks) -> None:
        """Select the next attack algorithm."""
        self._current_attack = min(self._current_attack + 1, len(self._attacks) - 1)

    def __repr__(self: Attacks) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Attacks (Attack algorithm count={len(self._attacks)})>"


# TODO: Look into attack proxy idea where c++ does heavy lifting and returns optional
#  tuple of attack enum and dict with data needed by python to create sprite object if
#  needed or some other implementation
