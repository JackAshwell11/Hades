"""Manages the different attributes available."""
from __future__ import annotations

# Builtin
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hades.game_objects.enums import EntityAttributeData

__all__ = ("StatusEffect",)


@dataclass(slots=True)
class StatusEffect:
    value: float
    duration: float


class EntityAttribute:
    """Represents an attribute that is part of an entity.

    Attributes
    ----------
    applied_status_effect: StatusEffect | None
        The currently applied status effect.
    """

    __slots__ = (
        "_value",
        "_max_value",
        "applied_status_effect",
        "attribute_data",
        "variable",
        "maximum",
    )

    def __init__(
        self: EntityAttribute,
        level: int,
        attribute_data: EntityAttributeData,
    ) -> None:
        """Initialise the object.

        Parameters
        ----------
        level: int
            The level to initialise the attribute at. These should start at 0 and
            increase over the game time.
        attribute_data: EntityAttributeData
            The data for this attribute.
        """
        self._value: float = attribute_data.increase(level)
        self._max_value: float = (
            self._value
            if attribute_data.variable and attribute_data.maximum
            else float("inf")
        )
        self.attribute_data: EntityAttributeData = attribute_data
        self.applied_status_effect: StatusEffect | None = None

    @property
    def value(self: EntityAttribute) -> float:
        """Get the attribute's value.

        Returns
        -------
        float
            The attribute's value.
        """
        return self._value

    @value.setter
    def value(self: EntityAttribute, value: float) -> None:
        """Set the attribute's value if possible.

        Parameters
        ----------
        value: float
            The new attribute value.

        Raises
        ------
        ValueError
            This attribute's value cannot be set.
        """
        # Check if the attribute value can be changed
        if not self.variable:
            raise ValueError("This attribute's value cannot be set.")

        # Update the attribute value with the new value
        self._value = value

        # Check if the attribute value exceeds the max. If so, set it to the max
        if self.value > self.max_value:
            self._value = self._max_value

    @property
    def max_value(self: EntityAttribute) -> float:
        """Get the attribute's max value.

        If this is -1, then the attribute is not variable.

        Returns
        -------
        float
            The attribute's max value.
        """
        return self._max_value

    def apply_status_effect(self: EntityAttribute) -> None:
        """Apply a status effect to the attribute if possible."""
        raise NotImplementedError

    def update_status_effect(self: EntityAttribute) -> None:
        """Update the currently applied status effect."""
        raise NotImplementedError

    def remove_status_effect(self: EntityAttribute) -> None:
        """Remove the currently applied status effect from the attribute."""
        raise NotImplementedError

    def apply_instant_effect(self: EntityAttribute) -> bool:
        """Apply an instant effect to the attribute if possible."""
        raise NotImplementedError

    def __repr__(self: EntityAttribute) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<EntityAttribute (Value={self.value})>"


class Health:
    """Allows a game object to have health.

    Attributes
    ----------
    health: EntityAttribute
        The game object's health attribute.
    """

    def __init__(self: Health, health_data: EntityAttributeData) -> None:
        """Initialise the object.

        Parameters
        ----------
        health_data: EntityAttributeData
            The health data for this component.
        """
        self.health: EntityAttribute = EntityAttribute(0, health_data)


class Armour:
    """Allows a game object to have armour.

    Attributes
    ----------
    armour: EntityAttribute
        The game object's armour attribute.
    """

    def __init__(self: Armour, armour_data: EntityAttributeData) -> None:
        """Initialise the object.

        Parameters
        ----------
        armour_data: EntityAttributeData
            The armour data for this component.
        """
        self.armour: EntityAttribute = EntityAttribute(0, armour_data)


class SpeedMultiplier:
    """Allows a game object to have a speed multiplier.

    Attributes
    ----------
    speed_multiplier: EntityAttribute
        The game object's speed multiplier attribute.
    """

    def __init__(self: Armour, speed_multiplier_data: EntityAttributeData) -> None:
        """Initialise the object.

        Parameters
        ----------
        speed_multiplier_data: EntityAttributeData
            The speed multiplier data for this component.
        """
        self.speed_multiplier: EntityAttribute = EntityAttribute(
            0, speed_multiplier_data
        )


class ArmourRegenCooldown:
    """Allows a game object to have an armour regen cooldown.

    Attributes
    ----------
    armour_regen_cooldown: EntityAttribute
        The game object's armour regen cooldown attribute.
    """

    def __init__(self: Armour, armour_regen_cooldown_data: EntityAttributeData) -> None:
        """Initialise the object.

        Parameters
        ----------
        armour_regen_cooldown_data: EntityAttributeData
            The armour regen cooldown data for this component.
        """
        self.armour_regen_cooldown: EntityAttribute = EntityAttribute(
            0, armour_regen_cooldown_data
        )


class FireRatePenalty:
    """Allows a game object to have a fire rate penalty.

    Attributes
    ----------
    fire_rate_penalty: EntityAttribute
        The game object's fire rate penalty attribute.
    """

    def __init__(self: Armour, fire_rate_penalty_data: EntityAttributeData) -> None:
        """Initialise the object.

        Parameters
        ----------
        fire_rate_penalty_data: EntityAttributeData
            The fire rate penalty data for this component.
        """
        self.fire_rate_penalty: EntityAttribute = EntityAttribute(
            0, fire_rate_penalty_data
        )


class Money:
    """Allows a game object to have money.

    Attributes
    ----------
    money: EntityAttribute
        The game object's money attribute.
    """

    def __init__(self: Money, money_data: EntityAttributeData) -> None:
        """Initialise the object.

        Parameters
        ----------
        money_data: EntityAttributeData
            The money data for this component.
        """
        self.money: EntityAttribute = EntityAttribute(0, money_data)
