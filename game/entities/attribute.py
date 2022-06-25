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


class AttributeBase:
    pass


class UpgradableAttribute(AttributeBase):
    pass


class StatusEffectAttribute(AttributeBase):
    pass


class VariableAttribute(AttributeBase):
    pass


class EntityAttribute:
    def __init__(self, owner: Entity, attribute_data: EntityAttributeData) -> None:
        self.upgradable: UpgradableAttribute | None = (
            UpgradableAttribute() if attribute_data.upgradable else None
        )
        self.status_effect: StatusEffectAttribute | None = (
            StatusEffectAttribute() if attribute_data.status_effect else None
        )
        self.variable: VariableAttribute | None = (
            VariableAttribute() if attribute_data.variable else None
        )
