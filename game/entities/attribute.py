"""TO DO!"""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.game_object import EntityAttributeType

if TYPE_CHECKING:
    from collections.abc import Callable

    from game.constants.game_object import (
        EntityAttributeData,
        EntityAttributeSectionType,
        StatusEffectData,
    )
    from game.entities.base import Entity
    from game.entities.player import Player

__all__ = (
    "EntityAttribute",
    "UpgradablePlayerSection",
)

# Get the logger
logger = logging.getLogger(__name__)


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


class UpgradablePlayerSection:
    """Represents a player attribute section that can be upgraded.

    Parameters
    ----------
    player: Player
        The reference to the player object used for upgrading the entity attributes.
    attribute_section_type: EntityAttributeSectionType
        The attribute section type this upgradable section represents.
    cost_function: Callable[[int], float]
        The cost function for this upgradable section used for calculating the next
        level's cost.
    current_level: int
        The current level for this upgradable section.
    """

    __slots__ = (
        "player",
        "attribute_section_type",
        "cost_function",
        "current_level",
    )

    def __init__(
        self,
        player: Player,
        attribute_section_type: EntityAttributeSectionType,
        cost_function: Callable[[int], float],
        current_level: int,
    ) -> None:
        self.player: Player = player
        self.attribute_section_type: EntityAttributeSectionType = attribute_section_type
        self.cost_function: Callable[[int], float] = cost_function
        self.current_level: int = current_level

    def __repr__(self) -> str:
        return (
            "<UpgradablePlayerSection (Attribute section"
            f" type={self.attribute_section_type}) (Current level={self.current_level})"
            f" (Level limit={self.level_limit})>"
        )

    @property
    def next_level_cost(self) -> int:
        """Gets the cost for the next level.

        Returns
        -------
        int
            The next level cost.
        """
        return round(self.cost_function(self.current_level))

    @property
    def level_limit(self) -> int:
        """Gets the maximum level for the player's upgrades.

        Returns
        -------
        int
            The maximum level for the player's upgrades.
        """
        return self.player.entity_data.level_limit

    def upgrade_section(self) -> bool:
        """Upgrades the player section if possible.

        Returns
        -------
        bool
            Whether the upgrade was successful or not.
        """
        # Check if the player has enough money and the current level is below the limit
        if (
            self.player.money.value < self.next_level_cost
            or self.current_level >= self.level_limit
        ):
            return False

        # Section upgrade is valid so subtract the cost from the player's money and
        # increment the current level
        self.player.money.value -= self.next_level_cost
        self.current_level += 1

        # Now upgrade each entity attribute
        logger.debug("Upgrading section %r", self.attribute_section_type)
        for (
            entity_attribute
        ) in self.attribute_section_type.value:  # type: EntityAttributeType
            # Calculate the diff between the current level and the next (this is
            # because some attribute may be variable or have status effects applied
            # to them)
            diff = self.cost_function(self.current_level) - self.cost_function(
                self.current_level - 1
            )

            # Apply that diff to the target entity attribute's value and max value
            # (if the attribute is variable)
            target_attribute = self.player.entity_state[entity_attribute]
            target_attribute._value += diff  # noqa
            if target_attribute.attribute_data.variable:
                target_attribute._max_value += diff  # noqa

            # If the entity attribute is health or armour, we need to update the
            # indicator bars
            if (
                entity_attribute is EntityAttributeType.HEALTH
                or entity_attribute is EntityAttributeType.ARMOUR
            ):
                self.player.update_indicator_bars()
            logger.debug("Upgraded attribute %r", entity_attribute)

        # Upgrade successful
        return True


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
        self._max_value: float = (
            self._value
            if attribute_data.variable and attribute_data.maximum
            else float("inf")
        )
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
        ValueError
            This attribute's value cannot be set.
        """
        # Check if the attribute value can be changed
        if not self.attribute_data.variable:
            raise ValueError("This attribute's value cannot be set.")

        # Update the attribute value with the new value
        self._value = value
        logger.debug("Set %r value for entity %r to %d", self, self.owner, self.value)

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
