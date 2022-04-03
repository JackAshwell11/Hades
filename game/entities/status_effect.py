from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from constants.entity import StatusEffectType

if TYPE_CHECKING:
    from entities.player import Player

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
    increase_amount: int
        The amount of increase that should be applied to the player temporarily.
    duration: float
        The duration the status effect should be applied for.
    original: int
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
        increase_amount: int,
        duration: float,
        original: int,
    ) -> None:
        self.player: Player = player
        self.effect_type: StatusEffectType = effect_type
        self.increase_amount: int = increase_amount
        self.duration: float = duration
        self.original: int = original
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

        # Apply the effect
        match self.effect_type:
            case StatusEffectType.HEALTH:
                self.player.health = new_value
                self.player.state_modifiers["bonus health"] = self.increase_amount
            case StatusEffectType.ARMOUR:
                self.player.armour = new_value
                self.player.state_modifiers["bonus armour"] = self.increase_amount
            case StatusEffectType.SPEED:
                pass
            case StatusEffectType.FIRE_RATE:
                pass

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
            # Get the current value
            current_value: float = -1
            match self.effect_type:
                case StatusEffectType.HEALTH:
                    current_value = self.player.health
                case StatusEffectType.ARMOUR:
                    current_value = self.player.armour
                case StatusEffectType.SPEED:
                    pass
                case StatusEffectType.FIRE_RATE:
                    pass

            # Check if the player needs to change at all
            if current_value > self.original:
                current_value = self.original

            # Apply the new change to the player
            match self.effect_type:
                case StatusEffectType.HEALTH:
                    self.player.state_modifiers["bonus health"] = 0
                    self.player.health = current_value
                case StatusEffectType.ARMOUR:
                    self.player.state_modifiers["bonus armour"] = 0
                    self.player.armour = current_value
                case StatusEffectType.SPEED:
                    pass
                case StatusEffectType.FIRE_RATE:
                    pass

            # Remove the status effect
            self.player.applied_effects.remove(self)
