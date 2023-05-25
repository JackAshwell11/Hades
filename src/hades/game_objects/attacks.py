"""Manages the different attack algorithms available."""
from __future__ import annotations

# Builtin
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

# Custom
from hades.constants import ARMOUR_REGEN_AMOUNT, ComponentType
from hades.game_objects.base import GameObjectComponent

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentData
    from hades.game_objects.system import ECS

__all__ = ("Attacker",)


class AttackBase(metaclass=ABCMeta):
    """The base class for all attack algorithms."""

    @abstractmethod
    def do_attack(self: AttackBase) -> None:
        """Perform an attack on other game objects."""


class Attacker(GameObjectComponent):
    """Allows a game object to attack and be attacked by other game objects.

    Attributes:
        do_regen: Whether the game object can regenerate armour or not.
        in_combat: Whether the game object is in combat or not.
        time_since_last_attack: The time since the last attack.
        time_since_armour_regen: The time since the game object last regenerated armour.
    """

    __slots__ = (
        "do_regen",
        "in_combat",
        "time_since_last_attack",
        "time_since_armour_regen",
    )

    # Class variables
    component_type: ComponentType = ComponentType.ARMOUR

    def __init__(
        self: Attacker,
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
        self.do_regen: bool = component_data["armour_regen"]
        self.in_combat: bool = False
        self.time_since_last_attack: float = 0
        self.time_since_armour_regen: float = 0

    def regenerate_armour(self: Attacker, delta_time: float) -> None:
        """Regenerate the game object's armour if possible.

        Args:
            delta_time: Time interval since the last time the function was called.
        """
        # Check if the game object's armour can regenerate
        if not self.do_regen:
            return

        # Regenerate the game object's armour
        if (
            self.time_since_armour_regen
            >= self.system.get_component_for_game_object(
                self.game_object_id,
                ComponentType.ARMOUR_REGEN_COOLDOWN,
            ).value
        ):
            self.system.get_component_for_game_object(
                self.game_object_id,
                ComponentType.ARMOUR,
            ).value += ARMOUR_REGEN_AMOUNT
            self.time_since_armour_regen = 0
        else:
            self.time_since_armour_regen += delta_time

    def do_ranged_attack(self: Attacker) -> None:
        """Perform a ranged attack in the direction the entity is facing."""
        raise NotImplementedError

    def do_melee_attack(self: Attacker) -> None:
        """Perform a melee attack in the direction the entity is facing."""
        raise NotImplementedError

    def do_area_of_effect_attack(self: Attacker) -> None:
        """Perform an area of effect attack around the entity."""
        raise NotImplementedError


# TODO: Can have attacker component with attack method in it. The component will have
#  the counters and combat booleans along with armour regeneration and the weapon cycler

# TODO: Maybe rearrange all namedtuples and stuff, idk
