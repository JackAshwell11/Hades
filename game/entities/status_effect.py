from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.consumable import StatusEffectType

if TYPE_CHECKING:
    from game.entities.player import Player

# Get the logger
logger = logging.getLogger(__name__)


class StatusEffect:
    """
    Stores the state of a currently applied status effect.

    Parameters
    ----------
    player: Player
        The reference to the player object
    effect_type: StatusEffectType
        The status effect type that is being applied to the player.
    increase_amount: float
        The amount of increase that should be applied to the player temporarily.
    duration: int
        The duration the status effect should be applied for.
    original: float
        The original value of the variable which is being changed.

    Attributes
    ----------
    time_counter: float
        The time counter for the status effect.
    """

    def __init__(
        self,
        player: Player,
        effect_type: StatusEffectType,
        increase_amount: float,
        duration: float,
        original: float,
    ) -> None:
        self.player: Player = player
        self.effect_type: StatusEffectType = effect_type
        self.increase_amount: float = increase_amount
        self.duration: float = duration
        self.original: float = original
        self.time_counter: float = 0

    def __repr__(self) -> str:
        return (
            f"<StatusEffect (Effect type={self.effect_type.name}) (Increase"
            f" amount={self.increase_amount}) (Duration={self.duration})"
            f" (Original={self.original})>"
        )

    def apply_effect(self) -> None:
        """Applies the status effect to the player."""
        # Get the new value
        new_value = self.original + self.increase_amount
        logger.info(
            f"Applying effect {self.effect_type} with amount {self.increase_amount} for"
            f" {self.duration} seconds"
        )

        # Apply the effect
        match self.effect_type:
            case StatusEffectType.HEALTH:
                self.player.health = int(new_value)
                self.player.max_health = int(
                    self.player.max_health + self.increase_amount
                )
            case StatusEffectType.ARMOUR:
                self.player.armour = int(new_value)
                self.player.max_armour = int(
                    self.player.max_armour + self.increase_amount
                )
            case StatusEffectType.SPEED:
                self.player.pymunk.max_velocity = new_value
            case StatusEffectType.FIRE_RATE:
                self.player.bonus_attack_cooldown = new_value

    def update(self, delta_time: float) -> None:
        """
        Updates a status effect checking if it has run out.

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
            if (
                self.effect_type is StatusEffectType.HEALTH
                or self.effect_type is StatusEffectType.ARMOUR
            ):
                # Get the current value
                current_value: float = -1
                match self.effect_type:
                    case StatusEffectType.HEALTH:
                        current_value = self.player.health
                    case StatusEffectType.ARMOUR:
                        current_value = self.player.armour

                # Check if the player needs to change at all
                if current_value > self.original:
                    current_value = self.original
            else:
                # Status effect is a speed or fire rate one so current_value isn't
                # needed
                current_value = -1

            # Apply the new change to the player
            match self.effect_type:
                case StatusEffectType.HEALTH:
                    self.player.health = int(current_value)
                    self.player.max_health = int(
                        self.player.max_health - self.increase_amount
                    )
                case StatusEffectType.ARMOUR:
                    self.player.armour = int(current_value)
                    self.player.max_armour = int(
                        self.player.max_armour - self.increase_amount
                    )
                case StatusEffectType.SPEED:
                    self.player.pymunk.max_velocity = self.original
                case StatusEffectType.FIRE_RATE:
                    self.player.bonus_attack_cooldown = self.original

            # Remove the status effect
            self.player.applied_effects.remove(self)
            logger.info(
                f"Removed effect {self.effect_type} setting value to {current_value}"
            )
