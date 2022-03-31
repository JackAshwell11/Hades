from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Custom
from constants.entity import StatusEffectType

if TYPE_CHECKING:
    from entities.player import Player


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
    duration: float
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
        self._match_status_effect(self.original + self.increase_amount)

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
                self._match_status_effect(self.original)

            # Remove the status effect
            self.player.applied_effects.remove(self)

    def _match_status_effect(self, value: float) -> None:
        """
        Matches the effect type to a player variable. This is only meant for internal
        use within this class.

        Parameters
        ----------
        value: float
            The value to change a player variable too.
        """
        match self.effect_type:
            case StatusEffectType.HEALTH:
                self.player.health = value
            case StatusEffectType.ARMOUR:
                self.player.armour = value
            case StatusEffectType.SPEED:
                pass
            case StatusEffectType.FIRE_RATE:
                pass
