"""Manages the different attack algorithms available."""
from __future__ import annotations

# Builtin
import math
from typing import TYPE_CHECKING, TypedDict

# Pip
from pymunk import Vec2d

# Custom
from hades.constants import (
    ATTACK_RANGE,
    BULLET_VELOCITY,
    DAMAGE,
    MELEE_ATTACK_OFFSET_LOWER,
    MELEE_ATTACK_OFFSET_UPPER,
)
from hades.game_objects.attributes import deal_damage
from hades.game_objects.base import AttackAlgorithms, ComponentType, GameObjectComponent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hades.game_objects.base import ComponentData
    from hades.game_objects.movements import PhysicsObject
    from hades.game_objects.system import ECS

__all__ = ("Attacks", "AttackResult")


class AttackResult(TypedDict, total=False):
    """Holds the result of an attack."""

    ranged_attack: tuple[Vec2d, float, float]


class Attacks(GameObjectComponent):
    """Allows a game object to attack other game objects."""

    __slots__ = ("_attacks", "_attack_state", "_steering_owner")

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
        self._attacks: Sequence[AttackAlgorithms] = component_data["enabled_attacks"]
        self._attack_state: int = 0
        self._steering_owner: PhysicsObject = (
            self.system.get_physics_object_for_game_object(self.game_object_id)
        )

    @property
    def attack_state(self: Attacks) -> int:
        """Get the index of the current attack algorithm.

        Returns:
            The index of the current attack algorithm.
        """
        return self._attack_state

    def _area_of_effect_attack(self: Attacks, targets: list[int]) -> None:
        """Perform an area of effect attack around the game object.

        Args:
            targets: The targets to attack.
        """
        # Find all targets that are within range and attack them
        for target in targets:
            if (
                self._steering_owner.position.get_distance(
                    self.system.get_physics_object_for_game_object(target).position,
                )
                <= ATTACK_RANGE
            ):
                deal_damage(target, self.system, DAMAGE)

    def _melee_attack(self: Attacks, targets: list[int]) -> None:
        """Perform a melee attack in the direction the game object is facing.

        Args:
            targets: The targets to attack.
        """
        # Calculate a vector that is perpendicular to the current rotation of the game
        # object
        physics_object = self.system.get_physics_object_for_game_object(
            self.game_object_id,
        )
        rotation = Vec2d(
            math.sin(math.radians(physics_object.rotation)),
            math.cos(math.radians(physics_object.rotation)),
        )

        # Find all targets that can be attacked
        for target in targets:
            # Calculate the angle between the current rotation of the game object and
            # the direction the target is in
            target_position = self.system.get_physics_object_for_game_object(
                target,
            ).position
            theta = (target_position - physics_object.position).get_angle_between(
                rotation,
            ) % (2 * math.pi)

            # Test if the target is within range and within the circle's sector
            if (
                physics_object.position.get_distance(target_position) <= ATTACK_RANGE
                and theta <= MELEE_ATTACK_OFFSET_LOWER
                or theta >= MELEE_ATTACK_OFFSET_UPPER
            ):
                deal_damage(target, self.system, DAMAGE)

    def _ranged_attack(self: Attacks) -> AttackResult:
        """Perform a ranged attack in the direction the game object is facing.

        Returns:
            The result of the attack.
        """
        # Calculate the bullet's angle of rotation
        physics_object = self.system.get_physics_object_for_game_object(
            self.game_object_id,
        )
        angle_radians = physics_object.rotation * math.pi / 180

        # Return the result of the attack
        return {
            "ranged_attack": (
                physics_object.position,
                math.cos(angle_radians) * BULLET_VELOCITY,
                math.sin(angle_radians) * BULLET_VELOCITY,
            ),
        }

    def do_attack(self: Attacks, targets: list[int]) -> AttackResult:
        """Perform the currently selected attack algorithm.

        Args:
            targets: The targets to attack.

        Returns:
            The result of the attack.
        """
        # Perform the attack on the targets
        match self._attacks[self._attack_state]:
            case AttackAlgorithms.AREA_OF_EFFECT_ATTACK:
                self._area_of_effect_attack(targets)
            case AttackAlgorithms.MELEE_ATTACK:
                self._melee_attack(targets)
            case AttackAlgorithms.RANGED_ATTACK:
                return self._ranged_attack()
            case _:  # pragma: no cover
                # This should never happen as all attacks are covered above
                raise ValueError

        # Return an empty result as no ranged attack was performed
        return {}

    def previous_attack(self: Attacks) -> None:
        """Select the previous attack algorithm."""
        self._attack_state = max(self._attack_state - 1, 0)

    def next_attack(self: Attacks) -> None:
        """Select the next attack algorithm."""
        self._attack_state = min(self._attack_state + 1, len(self._attacks) - 1)

    def __repr__(self: Attacks) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Attacks (Attack algorithm count={len(self._attacks)})>"
