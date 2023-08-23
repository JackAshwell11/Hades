"""Manages the game object attribute system and its various attributes."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, ClassVar

# Custom
from hades.constants import ARMOUR_REGEN_AMOUNT
from hades.game_objects.base import SystemBase
from hades.game_objects.components import (
    Armour,
    ArmourRegen,
    ArmourRegenCooldown,
    FireRatePenalty,
    Health,
    Money,
    MovementForce,
    StatusEffect,
    ViewDistance,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.components import (
        GameObjectAttributeBase,
    )


__all__ = ("ArmourRegenSystem", "GameObjectAttributeError", "GameObjectAttributeSystem")


class ArmourRegenSystem(SystemBase):
    """Provides facilities to manipulate armour regen components."""

    def update(self: ArmourRegenSystem, delta_time: float) -> None:
        """Process update logic for an armour regeneration component.

        Args:
            delta_time: The time interval since the last time the function was called.
        """
        for _, (
            armour,
            armour_regen,
            armour_regen_cooldown,
        ) in self.registry.get_components(Armour, ArmourRegen, ArmourRegenCooldown):
            armour_regen.time_since_armour_regen += delta_time
            if armour_regen.time_since_armour_regen >= armour_regen_cooldown.value:
                armour.value += ARMOUR_REGEN_AMOUNT
                armour_regen.time_since_armour_regen = 0


class GameObjectAttributeError(Exception):
    """Raised when there is an error with a game object attribute."""

    def __init__(self: GameObjectAttributeError, *, name: str, error: str) -> None:
        """Initialise the object.

        Args:
            name: The name of the game object attribute.
            error: The problem raised by the game object attribute.
        """
        super().__init__(f"The game object attribute `{name}` cannot {error}.")


class GameObjectAttributeSystem(SystemBase):
    """Provides facilities to manipulate game object attributes."""

    GAME_OBJECT_ATTRIBUTES: ClassVar[set[type[GameObjectAttributeBase]]] = {
        Armour,
        ArmourRegenCooldown,
        FireRatePenalty,
        Health,
        Money,
        MovementForce,
        ViewDistance,
    }

    def update(self: GameObjectAttributeSystem, delta_time: float) -> None:
        """Process update logic for a game object attribute.

        Args:
            delta_time: The time since the last update.
        """
        # Loop over all game object attributes and update them
        for game_object_attribute_type in self.GAME_OBJECT_ATTRIBUTES:
            for _, (game_object_attribute,) in self.registry.get_components(
                game_object_attribute_type,
            ):
                # Update the status effect if one is applied
                if status_effect := game_object_attribute.applied_status_effect:
                    status_effect.time_counter += delta_time
                    if status_effect.time_counter >= status_effect.duration:
                        game_object_attribute.value = min(
                            game_object_attribute.value,
                            status_effect.original_value,
                        )
                        game_object_attribute.max_value = (
                            status_effect.original_max_value
                        )
                        game_object_attribute.applied_status_effect = None

    def upgrade(
        self: GameObjectAttributeSystem,
        game_object_id: int,
        game_object_attribute_type: type[GameObjectAttributeBase],
        increase: Callable[[int], float],
    ) -> bool:
        """Upgrade the game object attribute to the next level if possible.

        Args:
            game_object_id: The ID of the game object to upgrade the game object
            attribute for.
            game_object_attribute_type: The type of game object attribute to upgrade.
            increase: The exponential lambda function which calculates the next level's
                value based on the current level.

        Returns:
            Whether the game object attribute upgrade was successful or not.

        Raises:
            GameObjectAttributeError: The game object attribute cannot be upgraded.
        """
        # Check if the attribute can be upgraded
        game_object_attribute = self.registry.get_component_for_game_object(
            game_object_id,
            game_object_attribute_type,
        )
        if not game_object_attribute.upgradable:
            raise GameObjectAttributeError(
                name=game_object_attribute.__class__.__name__,
                error="be upgraded",
            )

        # Check if the current level is below the level limit
        if game_object_attribute.current_level >= game_object_attribute.level_limit:
            return False

        # Upgrade the attribute based on the difference between the current level and
        # the next
        diff = increase(game_object_attribute.current_level + 1) - increase(
            game_object_attribute.current_level,
        )
        game_object_attribute.max_value += diff
        game_object_attribute.current_level += 1
        game_object_attribute.value += diff
        return True

    def apply_instant_effect(
        self: GameObjectAttributeSystem,
        game_object_id: int,
        game_object_attribute_type: type[GameObjectAttributeBase],
        increase: Callable[[int], float],
        level: int,
    ) -> bool:
        """Apply an instant effect to the game object attribute if possible.

        Args:
            game_object_id: The game object ID to upgrade the attribute of.
            game_object_attribute_type: The type of game object attribute to apply an
            instant effect to.
            increase: The exponential lambda function which calculates the next level's
                value based on the current level.
            level: The level to initialise the instant effect at.

        Returns:
            Whether the instant effect could be applied or not.

        Raises:
            GameObjectAttributeError: The game object attribute cannot have an instant
            effect.
        """
        # Check if the attribute can have an instant effect
        game_object_attribute = self.registry.get_component_for_game_object(
            game_object_id,
            game_object_attribute_type,
        )
        if not game_object_attribute.instant_effect:
            raise GameObjectAttributeError(
                name=game_object_attribute.__class__.__name__,
                error="have an instant effect",
            )

        # Check if the attribute's value is already at max
        if game_object_attribute.value == game_object_attribute.max_value:
            return False

        # Add the instant effect to the attribute
        game_object_attribute.value += increase(level)
        return True

    def apply_status_effect(
        self: GameObjectAttributeSystem,
        game_object_id: int,
        game_object_attribute_type: type[GameObjectAttributeBase],
        status_effect: tuple[Callable[[int], float], Callable[[int], float]],
        level: int,
    ) -> bool:
        """Apply a status effect to the attribute if possible.

        Args:
            game_object_id: The game object ID to upgrade the attribute of.
            game_object_attribute_type: The type of game object attribute to apply a
                status effect to.
            status_effect: The exponential lambda functions which calculate the next
                level's value and duration based on the current level.
            level: The level to initialise the status effect at.

        Returns:
            Whether the status effect could be applied or not.

        Raises:
            GameObjectAttributeError: The game object attribute cannot have a status
            effect.
        """
        # Check if the attribute can have a status effect
        game_object_attribute = self.registry.get_component_for_game_object(
            game_object_id,
            game_object_attribute_type,
        )
        if not game_object_attribute.status_effect:
            raise GameObjectAttributeError(
                name=game_object_attribute.__class__.__name__,
                error="have a status effect",
            )

        # Check if the attribute already has a status effect applied
        if game_object_attribute.applied_status_effect:
            return False

        # Apply the status effect to this attribute
        increase, duration = status_effect
        game_object_attribute.applied_status_effect = StatusEffect(
            increase(level),
            duration(level),
            game_object_attribute.value,
            game_object_attribute.max_value,
        )
        game_object_attribute.max_value += (
            game_object_attribute.applied_status_effect.value
        )
        game_object_attribute.value += game_object_attribute.applied_status_effect.value
        return True

    def deal_damage(
        self: GameObjectAttributeSystem,
        game_object_id: int,
        damage: int,
    ) -> None:
        """Deal damage to the game object.

        Args:
            game_object_id: The game object ID.
            damage: The damage that should be dealt to the game object.
        """
        # Damage the armour and carry over the extra damage to the health
        health, armour = (
            self.registry.get_component_for_game_object(game_object_id, Health),
            self.registry.get_component_for_game_object(game_object_id, Armour),
        )
        health.value -= max(damage - armour.value, 0)
        armour.value -= damage
