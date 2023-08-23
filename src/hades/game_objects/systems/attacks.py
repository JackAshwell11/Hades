"""Manages the attack system and its various attack algorithms."""
from __future__ import annotations

# Builtin
import math
from typing import TypedDict

# Custom
from hades.constants import (
    ATTACK_RANGE,
    BULLET_VELOCITY,
    DAMAGE,
    MELEE_ATTACK_OFFSET_LOWER,
    MELEE_ATTACK_OFFSET_UPPER,
)
from hades.game_objects.base import AttackAlgorithms, SystemBase
from hades.game_objects.components import Attacks
from hades.game_objects.steering import Vec2d
from hades.game_objects.systems.attributes import GameObjectAttributeSystem

__all__ = ("AttackResult", "AttackSystem")


class AttackResult(TypedDict, total=False):
    """Holds the result of an attack."""

    ranged_attack: tuple[Vec2d, float, float]


class AttackSystem(SystemBase):
    """Provides facilities to manipulate attack components."""

    def _area_of_effect_attack(
        self: AttackSystem,
        current_position: Vec2d,
        targets: list[int],
    ) -> None:
        """Perform an area of effect attack around the game object.

        Args:
            current_position: The current position of the game object.
            targets: The targets to attack.
        """
        # Find all targets that are within range and attack them
        for target in targets:
            if (
                current_position.get_distance_to(
                    self.registry.get_kinematic_object_for_game_object(target).position,
                )
                <= ATTACK_RANGE
            ):
                self.registry.get_system(GameObjectAttributeSystem).deal_damage(
                    target,
                    DAMAGE,
                )

    def _melee_attack(
        self: AttackSystem,
        current_position: Vec2d,
        current_rotation: float,
        targets: list[int],
    ) -> None:
        """Perform a melee attack in the direction the game object is facing.

        Args:
            current_position: The current position of the game object.
            current_rotation: The current rotation of the game object in radians.
            targets: The targets to attack.
        """
        # Calculate a vector that is perpendicular to the current rotation of the game
        # object
        rotation = Vec2d(
            math.sin(current_rotation),
            math.cos(current_rotation),
        )

        # Find all targets that can be attacked
        for target in targets:
            # Calculate the angle between the current rotation of the game object and
            # the direction the target is in
            target_position = self.registry.get_kinematic_object_for_game_object(
                target,
            ).position
            theta = (target_position - current_position).get_angle_between(rotation)

            # Test if the target is within range and within the circle's sector
            if (
                current_position.get_distance_to(target_position) <= ATTACK_RANGE
                and theta <= MELEE_ATTACK_OFFSET_LOWER
                or theta >= MELEE_ATTACK_OFFSET_UPPER
            ):
                self.registry.get_system(GameObjectAttributeSystem).deal_damage(
                    target,
                    DAMAGE,
                )

    @staticmethod
    def _ranged_attack(
        current_position: Vec2d,
        current_rotation: float,
    ) -> AttackResult:
        """Perform a ranged attack in the direction the game object is facing.

        Args:
            current_position: The current position of the game object.
            current_rotation: The current rotation of the game object in radians.

        Returns:
            The result of the attack.
        """
        return {
            "ranged_attack": (
                current_position,
                math.cos(current_rotation) * BULLET_VELOCITY,
                math.sin(current_rotation) * BULLET_VELOCITY,
            ),
        }

    def do_attack(
        self: AttackSystem,
        game_object_id: int,
        targets: list[int],
    ) -> AttackResult:
        """Perform the currently selected attack algorithm.

        Args:
            game_object_id: The ID of the game object to perform the attack for.
            targets: The targets to attack.

        Returns:
            The result of the attack.
        """
        # Perform the attack on the targets
        attacks, kinematic_object = self.registry.get_component_for_game_object(
            game_object_id,
            Attacks,
        ), self.registry.get_kinematic_object_for_game_object(game_object_id)
        match attacks.attacks[attacks.attack_state]:
            case AttackAlgorithms.AREA_OF_EFFECT_ATTACK:
                self._area_of_effect_attack(kinematic_object.position, targets)
            case AttackAlgorithms.MELEE_ATTACK:
                self._melee_attack(
                    kinematic_object.position,
                    math.radians(kinematic_object.rotation),
                    targets,
                )
            case AttackAlgorithms.RANGED_ATTACK:
                return self._ranged_attack(
                    kinematic_object.position,
                    math.radians(kinematic_object.rotation),
                )
            case _:  # pragma: no cover
                # This should never happen as all attacks are covered above
                raise ValueError

        # Return an empty result as no ranged attack was performed
        return {}

    def previous_attack(self: AttackSystem, game_object_id: int) -> None:
        """Select the previous attack algorithm.

        Args:
            game_object_id: The ID of the game object to select the previous attack for.
        """
        attacks = self.registry.get_component_for_game_object(game_object_id, Attacks)
        attacks.attack_state = max(attacks.attack_state - 1, 0)

    def next_attack(self: AttackSystem, game_object_id: int) -> None:
        """Select the next attack algorithm.

        Args:
            game_object_id: The ID of the game object to select the previous attack for.
        """
        attacks = self.registry.get_component_for_game_object(game_object_id, Attacks)
        attacks.attack_state = min(attacks.attack_state + 1, len(attacks.attacks) - 1)
