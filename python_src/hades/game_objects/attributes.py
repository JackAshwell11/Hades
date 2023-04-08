"""Manages the entity's attributes and their various properties."""
from __future__ import annotations

# Builtin
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Custom
from hades.constants.constructors import INDICATOR_BAR_TYPES
from hades.constants.game_objects import EntityAttributeType, StatusEffectType

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.constants.game_objects import (
        EntityAttributeData,
        EntityAttributeSectionType,
        InstantData,
        StatusEffectData,
    )
    from hades.game_objects.base import Entity
    from hades.game_objects.players import Player

__all__ = (
    "EntityAttribute",
    "UpgradablePlayerSection",
)

# Get the logger
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StatusEffect:
    """Represents a status effect that can be applied to an entity attribute.

    entity_attribute: EntityAttribute
        The entity attribute to apply the status effect too.
    status_effect_type: StatusEffectType
        The status effect type that this object represents.
    value: float
        The value that should be applied to the entity temporarily.
    duration: float
        The duration the status effect should be applied for.
    original: float
        The original value of the variable which is being changed.
    time_counter: float
        The time counter for the status effect.
    """

    entity_attribute: EntityAttribute
    status_effect_type: StatusEffectType
    value: float
    duration: float
    original: float = field(init=False)
    time_counter: float = field(init=False)

    def __post_init__(self) -> None:
        self.original = self.entity_attribute.value
        self.time_counter = 0


class UpgradablePlayerSection:
    """Represents a player attribute section that can be upgraded.

    Parameters
    ----------
    player: Player
        The reference to the player object used for upgrading the entity attributes.
    section_type: EntityAttributeSectionType
        The attribute section type this upgradable section represents.
    cost_function: Callable[[int], float]
        The cost function for this upgradable section used for calculating the next
        level's cost.
    current_level: int
        The current level for this upgradable section.
    """

    __slots__ = (
        "player",
        "section_type",
        "cost_function",
        "current_level",
    )

    def __init__(
        self,
        player: Player,
        section_type: EntityAttributeSectionType,
        cost_function: Callable[[int], float],
        current_level: int,
    ) -> None:
        self.player: Player = player
        self.section_type: EntityAttributeSectionType = section_type
        self.cost_function: Callable[[int], float] = cost_function
        self.current_level: int = current_level

    @property
    def next_level_cost(self) -> int:
        """Get the cost for the next level.

        Returns
        -------
        int
            The next level cost.
        """
        return round(self.cost_function(self.current_level))

    @property
    def level_limit(self) -> int:
        """Get the maximum level for the player's upgrades.

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
        logger.debug("Upgrading section %r", self.section_type)
        for entity_attribute in self.section_type.value:
            # Calculate the diff between the current level and the next (this is
            # because some attribute may be variable or have status effects applied
            # to them)
            diff = self.cost_function(self.current_level) - self.cost_function(
                self.current_level - 1,
            )

            # Apply that diff to the target entity attribute's value and max value
            # (if the attribute is variable)
            self.player.entity_state[entity_attribute].upgrade(diff)

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

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return (
            "<UpgradablePlayerSection (Attribute section"
            f" type={self.section_type}) (Current level={self.current_level})"
            f" (Level limit={self.level_limit})>"
        )


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
        self,
        owner: Entity,
        attribute_data: EntityAttributeData,
        level: int,
    ) -> None:
        self.owner: Entity = owner
        self.attribute_data: EntityAttributeData = attribute_data
        self._value: float = attribute_data.increase(level)
        self._max_value: float = (
            self._value
            if attribute_data.variable and attribute_data.maximum
            else float("inf")
        )
        self.applied_status_effect: StatusEffect | None = None

    @property
    def value(self) -> float:
        """Get the attribute's value.

        Returns
        -------
        float
            The attribute's value.
        """
        return self._value

    @value.setter
    def value(self, value: float) -> None:
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
        """Get the attribute's max value.

        If this is -1, then the attribute is not variable.

        Returns
        -------
        float
            The attribute's max value.
        """
        return self._max_value

    def upgrade(self, diff: float) -> None:
        """Upgrades this attribute. This should only be called on the player entity.

        Parameters
        ----------
        diff: float
            The difference to add to the attribute to increase it's level.
        """
        self._value += diff
        if self.attribute_data.variable:
            self._max_value += diff

    def apply_status_effect(
        self,
        status_effect_data: StatusEffectData,
        level: int,
    ) -> None:
        """Apply a status effect to the attribute if possible.

        Parameters
        ----------
        status_effect_data: StatusEffectData
            The status effect data to apply.
        level: int
            The level to initialise the status effect at.
        """
        # Test if there is already a status effect applied and if the attribute can have
        # one applied to it
        if self.applied_status_effect and self.attribute_data.status_effect:
            return

        # Apply the status effect to this attribute
        logger.debug(
            "Applying status effect %r to %r",
            status_effect_data.status_type,
            self,
        )
        new_status_effect = StatusEffect(
            self,
            status_effect_data.status_type,
            status_effect_data.increase(level),
            status_effect_data.duration(level),
        )
        self.applied_status_effect = new_status_effect
        self._value += new_status_effect.value
        if self.attribute_data.variable:
            self._max_value += new_status_effect.value

        # Apply custom status effect application logic
        if new_status_effect.status_effect_type is StatusEffectType.SPEED:
            new_value = new_status_effect.original + new_status_effect.value
            self.owner.pymunk.max_horizontal_velocity = new_value
            self.owner.pymunk.max_vertical_velocity = new_value
        elif new_status_effect.status_effect_type.value in INDICATOR_BAR_TYPES:
            self.owner.update_indicator_bars()

    def update_status_effect(self, delta_time: float) -> None:
        """Update the currently applied status effect.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Test if there isn't a status effect already applied
        if not self.applied_status_effect:
            return

        # Update the time counter
        current_status_effect = self.applied_status_effect
        current_status_effect.time_counter += delta_time

        # Apply custom status effect update logic
        if self.applied_status_effect.status_effect_type.value in INDICATOR_BAR_TYPES:
            self.owner.update_indicator_bars()

        # Check if we need to remove the status effect
        if current_status_effect.time_counter >= current_status_effect.duration:
            self.remove_status_effect()

    def remove_status_effect(self) -> None:
        """Remove the currently applied status effect from the attribute."""
        # Test if there isn't a status effect already applied
        if not self.applied_status_effect:
            return

        # Remove the status effect from this attribute while checking if the current
        # value is bigger than the original value. If so, we need to restore the
        # original value
        current_status_effect = self.applied_status_effect
        logger.debug(
            "Removing status effect %r from %r",
            current_status_effect.status_effect_type,
            self,
        )
        current_value = self.value
        if current_value > current_status_effect.original:
            current_value = current_status_effect.original
        self._value = current_value
        if self.attribute_data.variable:
            self._max_value = self.value - current_status_effect.value

        # Apply custom status effect remove logic
        if current_status_effect.status_effect_type is StatusEffectType.SPEED:
            self.owner.pymunk.max_horizontal_velocity = current_status_effect.original
            self.owner.pymunk.max_vertical_velocity = current_status_effect.original
        elif current_status_effect.status_effect_type.value in INDICATOR_BAR_TYPES:
            self.owner.update_indicator_bars()

        # Clear the applied status effect
        self.applied_status_effect = None

    def apply_instant_effect(self, instant_data: InstantData, level: int) -> bool:
        """Apply an instant effect to the attribute if possible.

        Parameters
        ----------
        instant_data: InstantData
            The instant effect data to apply.
        level: int
            The level to initialise the instant effect at.

        Returns
        -------
        bool
            Whether the instant effect could be applied or not.
        """
        # Check if the attribute's value is already at max
        if self.value == self.max_value:
            logger.debug(
                "%r for entity %r is already at max so instant effect can't be used",
                self,
                self.owner,
            )
            return False

        # Add the instant effect to the attribute and check if the new value is over the
        # max
        self.value = self.value + instant_data.increase(level)
        if self.value > self.max_value:
            self.value = self.max_value
            logger.debug("Set %r to max", self)

        # Apply custom instant effect apply logic
        if instant_data.instant_type.value in INDICATOR_BAR_TYPES:
            self.owner.update_indicator_bars()

        # Instant effect successful
        return True

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<EntityAttribute (Value={self._value})>"
