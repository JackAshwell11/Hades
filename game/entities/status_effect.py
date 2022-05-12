from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.consumable import StatusEffectType

if TYPE_CHECKING:
    from game.entities.base import Entity

# Get the logger
logger = logging.getLogger(__name__)


class StatusEffect:
    """
    Stores the state of a currently applied status effect on an entity.

    Parameters
    ----------
    target: Entity
        The reference to the target entity object
    effect_type: StatusEffectType
        The status effect type that is being applied to the entity.
    value: float
        The value that should be applied to the entity temporarily.
    duration: int
        The duration the status effect should be applied for.

    Attributes
    ----------
    original: float
        The original value of the variable which is being changed.
    time_counter: float
        The time counter for the status effect.
    """

    __slots__ = (
        "target",
        "effect_type",
        "value",
        "duration",
        "original",
        "time_counter",
    )

    def __init__(
        self,
        target: Entity,
        effect_type: StatusEffectType,
        value: float,
        duration: float,
    ) -> None:
        self.target: Entity = target
        self.effect_type: StatusEffectType = effect_type
        self.value: float = value
        self.duration: float = duration
        self.original: float = -1
        self.time_counter: float = 0

    def __repr__(self) -> str:
        return (
            f"<StatusEffect (Effect type={self.effect_type.name}) (Value={self.value})"
            f" (Duration={self.duration})"
        )

    def apply_effect(self) -> None:
        """Applies the status effect to the entity."""
        # Apply the effect
        match self.effect_type:
            case StatusEffectType.HEALTH:
                self.original = self.target.health
                self.target.health = self.original + self.value
                self.target.max_health = self.target.max_health + self.value
            case StatusEffectType.ARMOUR:
                self.original = self.target.armour
                self.target.armour = self.original + self.value
                self.target.max_armour = self.target.max_armour + self.value
            case StatusEffectType.SPEED:
                self.original = self.target.max_velocity
                self.target.pymunk.max_velocity = self.original + self.value
            case StatusEffectType.FIRE_RATE:
                self.original = self.target.bonus_attack_cooldown
                self.target.bonus_attack_cooldown = self.original + self.value
        logger.info(
            f"Applying effect {self.effect_type} with amount {self.value} for"
            f" {self.duration} seconds to entity {self.target}"
        )

    def update(self, delta_time: float) -> None:
        """
        Updates a status effect while checking if it has run out or not.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the time counter
        self.time_counter += delta_time

        # Check if we need to remove the status effect
        if self.time_counter >= self.duration:
            # Check if the status effect is a health or armour one
            current_value: float = -1
            if self.effect_type in [StatusEffectType.HEALTH, StatusEffectType.ARMOUR]:
                # Get the current value
                match self.effect_type:
                    case StatusEffectType.HEALTH:
                        current_value = self.target.health
                    case StatusEffectType.ARMOUR:
                        current_value = self.target.armour

                # Check if the target needs to change it's state
                if current_value > self.original:
                    current_value = self.original

            # Apply the new change to the target
            match self.effect_type:
                case StatusEffectType.HEALTH:
                    self.target.health = current_value
                    self.target.max_health = self.target.max_health - self.value
                case StatusEffectType.ARMOUR:
                    self.target.armour = current_value
                    self.target.max_armour = self.target.max_armour - self.value
                case StatusEffectType.SPEED:
                    self.target.pymunk.max_velocity = self.original
                case StatusEffectType.FIRE_RATE:
                    self.target.bonus_attack_cooldown = self.original

            # Remove the status effect
            self.target.applied_effects.remove(self)
            logger.info(
                f"Removed effect {self.effect_type} from entity {self.target} setting"
                f" value to {current_value}"
            )
