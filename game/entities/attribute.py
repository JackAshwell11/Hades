"""TO DO!"""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.constants.entity import EntityAttributeData
    from game.entities.base import Entity

__all__ = ("EntityAttribute",)

# Get the logger
logger = logging.getLogger(__name__)


class StatusEffect:
    """"""

    __slots__ = ()

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<StatusEffect>"


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
        "_attribute_data",
        "applied_status_effect",
        "_value",
        "_max_value",
    )

    def __init__(
        self, owner: Entity, attribute_data: EntityAttributeData, level: int
    ) -> None:
        self.owner: Entity = owner
        self._attribute_data: EntityAttributeData = attribute_data
        self.applied_status_effect: StatusEffect | None = None
        self._value: float = attribute_data.increase(level)
        self._max_value: float = self._value if attribute_data.variable else -1

    def __repr__(self) -> str:
        return f"<EntityAttribute (Value={self.value})>"

    def __lt__(self, other: EntityAttribute | float) -> bool:
        return self.value < getattr(other, "value", other)

    def __le__(self, other: EntityAttribute | float) -> bool:
        return self.value <= getattr(other, "value", other)

    def __eq__(self, other: EntityAttribute | float) -> bool:
        return self.value == getattr(other, "value", other)

    def __ne__(self, other: EntityAttribute | float) -> bool:
        return self.value != getattr(other, "value", other)

    def __ge__(self, other: EntityAttribute | float) -> bool:
        return self.value >= getattr(other, "value", other)

    def __gt__(self, other: EntityAttribute | float) -> bool:
        return self.value > getattr(other, "value", other)

    @property
    def value(self) -> float:
        """Gets the attribute's value.

        Returns
        -------
        float
            The attribute's value.
        """
        return self._value

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

    def change_value(self, new_value: float) -> None:
        """Changes the attribute's value if possible.

        Parameters
        ----------
        new_value: float
            The new attribute value.
        """
        # Check if the attribute value can be changed
        if not self._attribute_data.variable:
            return

        # Update the attribute value with the new value
        self._value = new_value

        # Check if the attribute value exceeds the max. If so, set it to the max
        if self.value > self.max_value:
            self._value = self._max_value
            logger.debug("Set %r attribute %r to max", self.owner, self)

    def apply_status_effect(self) -> None:
        """"""
