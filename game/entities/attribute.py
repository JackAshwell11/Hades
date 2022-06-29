"""TO DO!"""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.constants.game_object import EntityAttributeData, StatusEffectData
    from game.entities.base import Entity

__all__ = ("EntityAttribute",)

# Get the logger
logger = logging.getLogger(__name__)


class VariableError(Exception):
    """Raised when a non-variable entity attribute's value is set."""


# class StatusEffect:
#     """Represents a status effect that can be applied to an entity attribute.
#
#     Parameters
#     ----------
#     entity_attribute: EntityAttribute
#         The entity attribute to apply the status effect too.
#     status_effect_type: StatusEffectType
#         The status effect type that this object represents.
#     value: float
#         The value that should be applied to the entity temporarily.
#     duration: float
#         The duration the status effect should be applied for.
#
#     Attributes
#     ----------
#     original: float
#         The original value of the variable which is being changed.
#     time_counter: float
#         The time counter for the status effect.
#     """
#
#     __slots__ = (
#         "entity_attribute",
#         "status_effect_type",
#         "value",
#         "duration",
#         "original",
#         "time_counter"
#     )
#
#     def __init__(
#         self, entity_attribute: EntityAttribute, status_effect_type: StatusEffectType, value: float, duration: float
#     ) -> None:
#         self.entity_attribute: EntityAttribute = entity_attribute
#         self.status_effect_type: StatusEffectType = status_effect_type
#         self.value: float = value
#         self.duration: float = duration
#         self.original: float = -1
#         self.time_counter: float = 0
#         self._apply_effect()
#
#     def __repr__(self) -> str:
#         return f"<StatusEffect (Value={self.value}) (Duration={self.duration})>"
#
#     def _apply_effect(self) -> None:
#         """Applies the effect to the entity attribute."""
#         # Apply the status effect to the target
#         logger.debug("Applying health effect to %r", self.entity_attribute)
#         self.original = self.entity_attribute.value
#         self.entity_attribute._value += self.value  # noqa
#         if self.entity_attribute.attribute_data.variable:
#             self.entity_attribute._max_value += self.value  # noqa
#
#     def _remove_effect(self) -> None:
#         """Removes the effect from the entity attribute."""
#         # Get the target's current value to determine if its state needs to change
#         logger.debug("Removing health effect from %r", self.target)
#
#     def update(self, delta_time: float) -> None:
#         """Updates the state of a status effect.
#
#         Parameters
#         ----------
#         delta_time: float
#             Time interval since the last time the function was called.
#         """
#         # Update the time counter
#         self.time_counter += delta_time
#
#         # Check if we need to remove the status effect
#         if self.time_counter >= self.duration:
#             self._remove_effect()


class EntityAttribute:
    """Represents an attribute that is part of an entity.

    Parameters
    ----------
    owner: Entity
        The reference to the entity object that owns this attribute.
    attribute_data: EntityAttributeData
        The data for this attribute.
    level: int
        The level to initialise the attribute at. These should start at 0 and increase
        over the game time.

    Attributes
    ----------
    applied_status_effect: StatusEffect | None
        The currently applied status effect.
    """

    __slots__ = (
        "owner",
        "attribute_data",
        "applied_status_effect",
        "_value",
        "_max_value",
    )

    def __init__(
        self, owner: Entity, attribute_data: EntityAttributeData, level: int
    ) -> None:
        self.owner: Entity = owner
        self.attribute_data: EntityAttributeData = attribute_data
        self._value: float = attribute_data.increase(level)
        self._max_value: float = self._value if attribute_data.variable else -1
        self.applied_status_effect = None

    def __repr__(self) -> str:
        return f"<EntityAttribute (Value={self.value})>"

    @property
    def value(self) -> float:
        """Gets the attribute's value.

        Returns
        -------
        float
            The attribute's value.
        """
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        """Sets the attribute's value if possible.

        Parameters
        ----------
        value: float
            The new attribute value.

        Raises
        ------
        VariableError
            This attribute's value cannot be set.
        """
        # Check if the attribute value can be changed
        if not self.attribute_data.variable:
            raise VariableError("This attribute's value cannot be set.")

        # Update the attribute value with the new value
        self._value = value

        # Check if the attribute value exceeds the max. If so, set it to the max
        if self.value > self.max_value:
            self._value = self._max_value
            logger.debug("Set %r attribute %r to max", self.owner, self)

    @property
    def max_value(self) -> float:
        """Gets the attribute's max value. If this is -1, then the attribute is not
        variable.

        Returns
        -------
        float
            The attribute's max value.
        """
        return self._max_value

    def apply_status_effect(
        self, status_effect_data: StatusEffectData, level: int
    ) -> None:
        """Applies a status effect to the attribute if possible.

        Parameters
        ----------
        status_effect_data: StatusEffectData
            The status effect data to apply.
        level: int
            The level to initialise the status effect at.
        """
        raise NotImplementedError
