"""TO DO!"""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.constants.entity import EntityAttributeData

__all__ = ("EntityAttribute",)

# Get the logger
logger = logging.getLogger(__name__)


class AttributeBase:
    pass


class UpgradableAttribute(AttributeBase):
    pass


class StatusEffectAttribute(AttributeBase):
    pass


class VariableAttribute(AttributeBase):
    pass


class EntityAttribute:
    """Represents an attribute that is part of an entity.

    Parameters
    ----------
    level: int
        The level to initialise the attribute at. These should start at 0 and increase
        over the game time.
    attribute_data: EntityAttributeData
        The data for this attribute.

    Attributes
    ----------
    upgradable: UpgradableAttribute | None
        Allows the attribute to be upgraded to the next level.
    status_effect: StatusEffectAttribute
        Allows the attribute to have a status effect applied to it.
    variable: VariableAttribute
        Allows the attribute to vary from its initial value.
    """

    __slots__ = (
        "_value",
        "_attribute_data",
        "upgradable",
        "status_effect",
        "variable",
    )

    def __init__(self, level: int, attribute_data: EntityAttributeData) -> None:
        self._value: float = attribute_data.increase(level)
        self._attribute_data: EntityAttributeData = attribute_data
        self.upgradable: UpgradableAttribute | None = (
            UpgradableAttribute() if attribute_data.upgradable else None
        )
        self.status_effect: StatusEffectAttribute | None = (
            StatusEffectAttribute() if attribute_data.status_effect else None
        )
        self.variable: VariableAttribute | None = (
            VariableAttribute() if attribute_data.variable else None
        )

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
