"""Manages the different status effects that can be applied to an entity."""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.game_object import StatusEffectType

if TYPE_CHECKING:
    from game.entities.base import Entity

__all__ = (
    "STATUS_EFFECTS",
    "StatusEffectBase",
    "create_status_effect",
)

# Get the logger
logger = logging.getLogger(__name__)


class StatusEffectBase:
    """The base class for all status effects.

    Parameters
    ----------
    target: Entity
        The reference to the target entity object.
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
        "value",
        "duration",
        "original",
        "time_counter",
    )

    status_effect_type: StatusEffectType | None = None

    def __init__(self, target: Entity, value: float, duration: float) -> None:
        self.target: Entity = target
        self.value: float = value
        self.duration: float = duration
        self.original: float = -1
        self.time_counter: float = 0

    def __repr__(self) -> str:
        return f"<StatusEffectBase (Value={self.value}) (Duration={self.duration})"

    def apply_effect(self) -> None:
        """Applies the status effect to the entity.

        Raises
        ------
        NotImplementedError
            The function is not implemented.
        """
        raise NotImplementedError

    def update(self, delta_time: float) -> None:
        """Updates the state of a status effect.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the time counter
        self.time_counter += delta_time

        # Check if we need to remove the status effect
        if self.time_counter >= self.duration:
            self.remove_effect()

    def remove_effect(self) -> None:
        """Removes the status effect from the entity.

        Raises
        ------
        NotImplementedError
            The function is not implemented.
        """
        raise NotImplementedError


class HealthStatusEffect(StatusEffectBase):
    """Represents a health status effect that temporarily boosts the target's health.

    Parameters
    ----------
    target: Entity
        The reference to the target entity object.
    value: float
        The value that should be applied to the entity temporarily.
    duration: int
        The duration the status effect should be applied for.
    """

    __slots__ = ()

    status_effect_type: StatusEffectType = StatusEffectType.HEALTH

    def __init__(self, target: Entity, value: float, duration: float) -> None:
        super().__init__(target, value, duration)

    def __repr__(self) -> str:
        return f"<HealthStatusEffect (Value={self.value}) (Duration={self.duration})"

    def apply_effect(self) -> None:
        """Applies the status effect to the target."""
        # Apply the status effect to the target
        logger.debug("Applying health effect to %r", self.target)
        self.original = self.target.health
        self.target.health = self.original + self.value
        self.target.max_health = self.target.max_health + self.value

    def update(self, delta_time: float) -> None:
        """Updates the state of a status effect.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the time counter
        self.time_counter += delta_time

        # Check if we need to remove the status effect
        if self.time_counter >= self.duration:
            self.remove_effect()

    def remove_effect(self) -> None:
        """Removes the status effect from the entity."""
        # Get the target's current value to determine if its state needs to change
        logger.debug("Removing health effect from %r", self.target)
        current_value: float = self.target.health
        if current_value > self.original:
            current_value = self.original

        # Apply the new change to the target and remove the status effect
        self.target.health = current_value
        self.target.max_health = self.target.max_health - self.value
        self.target.applied_effects.remove(self)


class ArmourStatusEffect(StatusEffectBase):
    """Represents an armour status effect that temporarily boosts the target's armour.

    Parameters
    ----------
    target: Entity
        The reference to the target entity object.
    value: float
        The value that should be applied to the entity temporarily.
    duration: int
        The duration the status effect should be applied for.
    """

    __slots__ = ()

    status_effect_type: StatusEffectType = StatusEffectType.ARMOUR

    def __init__(self, target: Entity, value: float, duration: float) -> None:
        super().__init__(target, value, duration)

    def __repr__(self) -> str:
        return f"<ArmourStatusEffect (Value={self.value}) (Duration={self.duration})"

    def apply_effect(self) -> None:
        """Applies the status effect to the target."""
        # Apply the status effect to the target
        logger.debug("Applying armour effect to %r", self.target)
        self.original = self.target.armour
        self.target.armour = self.original + self.value
        self.target.max_armour = self.target.max_armour + self.value

    def update(self, delta_time: float) -> None:
        """Updates the state of a status effect.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the time counter
        self.time_counter += delta_time

        # Check if we need to remove the status effect
        if self.time_counter >= self.duration:
            self.remove_effect()

    def remove_effect(self) -> None:
        """Removes the status effect from the entity."""
        # Get the target's current value to determine if its state needs to change
        logger.debug("Removing armour effect from %r", self.target)
        current_value: float = self.target.armour
        if current_value > self.original:
            current_value = self.original

        # Apply the new change to the target and remove the status effect
        self.target.armour = current_value
        self.target.max_armour = self.target.max_armour - self.value
        self.target.applied_effects.remove(self)


class SpeedStatusEffect(StatusEffectBase):
    """Represents a speed status effect that temporarily boosts the target's speed.

    Parameters
    ----------
    target: Entity
        The reference to the target entity object.
    value: float
        The value that should be applied to the entity temporarily.
    duration: int
        The duration the status effect should be applied for.
    """

    __slots__ = ()

    status_effect_type: StatusEffectType = StatusEffectType.SPEED

    def __init__(self, target: Entity, value: float, duration: float) -> None:
        super().__init__(target, value, duration)

    def __repr__(self) -> str:
        return f"<SpeedStatusEffect (Value={self.value}) (Duration={self.duration})"

    def apply_effect(self) -> None:
        """Applies the status effect to the target."""
        # Apply the status effect to the target
        logger.debug("Applying speed effect to %r", self.target)
        self.original = self.target.max_velocity
        self.target.pymunk.max_velocity = self.original + self.value

    def update(self, delta_time: float) -> None:
        """Updates the state of a status effect.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the time counter
        self.time_counter += delta_time

        # Check if we need to remove the status effect
        if self.time_counter >= self.duration:
            self.remove_effect()

    def remove_effect(self) -> None:
        """Removes the status effect from the entity."""
        # Restore the original value and remove the status effect
        logger.debug("Removing speed effect from %r", self.target)
        self.target.pymunk.max_velocity = self.original
        self.target.applied_effects.remove(self)


class FireRateStatusEffect(StatusEffectBase):
    """Represents a fire rate status effect that temporarily boosts the target's fire
    rate.

    Parameters
    ----------
    target: Entity
        The reference to the target entity object.
    value: float
        The value that should be applied to the entity temporarily.
    duration: int
        The duration the status effect should be applied for.
    """

    __slots__ = ()

    status_effect_type: StatusEffectType = StatusEffectType.FIRE_RATE

    def __init__(self, target: Entity, value: float, duration: float) -> None:
        super().__init__(target, value, duration)

    def __repr__(self) -> str:
        return f"<FireRateStatusEffect (Value={self.value}) (Duration={self.duration})"

    def apply_effect(self) -> None:
        """Applies the status effect to the target."""
        # Apply the status effect to the target
        logger.debug("Applying fire rate effect to %r", self.target)
        self.original = self.target.bonus_attack_cooldown
        self.target.bonus_attack_cooldown = self.original + self.value

    def update(self, delta_time: float) -> None:
        """Updates the state of a status effect.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Update the time counter
        self.time_counter += delta_time

        # Check if we need to remove the status effect
        if self.time_counter >= self.duration:
            self.remove_effect()

    def remove_effect(self) -> None:
        """Removes the status effect from the entity."""
        # Restore the original value and remove the status effect
        logger.debug("Removing fire rate effect from %r", self.target)
        self.target.bonus_attack_cooldown = self.original
        self.target.applied_effects.remove(self)


STATUS_EFFECTS = {
    StatusEffectType.HEALTH: HealthStatusEffect,
    StatusEffectType.ARMOUR: ArmourStatusEffect,
    StatusEffectType.SPEED: SpeedStatusEffect,
    StatusEffectType.FIRE_RATE: FireRateStatusEffect,
}


def create_status_effect(
    status_effect_type: StatusEffectType, target: Entity, value: float, duration: float
) -> StatusEffectBase:
    """Determines which status effect class should be initialised based on a given
    status effect type.

    Parameters
    ----------
    status_effect_type: StatusEffectType
        The status effect to create.
    target: Entity
        The reference to the target entity object.
    value: float
        The value that should be applied to the entity temporarily.
    duration: int
        The duration the status effect should be applied for.
    """
    # Get the status effect class type which manages the given status effect
    cls = STATUS_EFFECTS[status_effect_type]
    logger.debug(
        "Selected status effect %r for status effect type %r", cls, status_effect_type
    )

    # Initialise the class with the given parameters
    return cls(target, value, duration)
