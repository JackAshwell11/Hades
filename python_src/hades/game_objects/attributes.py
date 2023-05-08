"""Manages the different game object attributes available."""
from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

# Custom
from hades.game_objects.base import ComponentData, ComponentType, GameObjectComponent

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = (
    "Armour",
    "ArmourRegenCooldown",
    "EntityAttributeBase",
    "EntityAttributeError",
    "FireRatePenalty",
    "Health",
    "Money",
    "MovementForce",
    "ViewDistance",
)


# Define a generic type for the keyword arguments
KW = TypeVar("KW")


class EntityAttributeError(Exception):
    """Raised when there is an error with an entity attribute."""

    def __init__(self: EntityAttributeError, *, name: str, error: str) -> None:
        """Initialise the object.

        Args:
            name: The name of the entity attribute.
            error: The problem raised by the entity attribute.
        """
        super().__init__(f"The entity attribute `{name}` cannot {error}.")


@dataclass(slots=True)
class StatusEffect:
    """Represents a status effect that can be applied to an entity attribute.

    value: The value that should be applied to the entity temporarily.
    duration: The duration the status effect should be applied for.
    original: The original value of the entity attribute which is being changed.
    original_max_value: The original maximum value of the entity attribute which is
        being changed.
    time_counter: The time counter for the status effect.
    """

    value: float
    duration: float
    original_value: float
    original_max_value: float
    time_counter: float = field(init=False)

    def __post_init__(self: StatusEffect) -> None:
        """Initialise the object after its initial initialization."""
        self.time_counter = 0


class EntityAttributeBase(GameObjectComponent):
    """The base class for all entity attributes."""

    __slots__ = (
        "_level_limit",
        "_value",
        "_max_value",
        "_applied_status_effect",
        "_current_level",
    )

    # Class variables
    instant_effect: bool = True
    maximum: bool = True
    status_effect: bool = True
    upgradable: bool = True

    def __init__(
        self: EntityAttributeBase,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            component_data: The data for the components.
        """
        super().__init__(component_data)
        initial_value, level_limit = component_data["attributes"][self.component_type]
        self._value: float = initial_value
        self._max_value: float = initial_value if self.maximum else float("inf")
        self._level_limit: int = level_limit
        self._current_level: int = 0
        self._applied_status_effect: StatusEffect | None = None

    @property
    def value(self: EntityAttributeBase) -> float:
        """Get the entity attribute's value.

        Returns:
            The entity attribute's value.
        """
        return self._value

    @value.setter
    def value(self: EntityAttributeBase, new_value: float) -> None:
        """Set the entity attribute's value.

        Args:
            new_value: The new entity attribute's value.
        """
        self._value = max(min(new_value, self._max_value), 0)

    @property
    def max_value(self: EntityAttributeBase) -> float:
        """Get the attribute's max value.

        If this is infinity, then the attribute does not have a maximum value

        Returns:
            The attribute's max value.
        """
        return self._max_value

    @property
    def current_level(self: EntityAttributeBase) -> int:
        """Get the attribute's current level.

        Returns:
            The attribute's current level.
        """
        return self._current_level

    @property
    def level_limit(self: EntityAttributeBase) -> int:
        """Get the attribute's level limit.

        If this is -1, then the attribute is not upgradable.

        Returns:
            The attribute's level limit.
        """
        return self._level_limit

    @property
    def applied_status_effect(self: EntityAttributeBase) -> StatusEffect | None:
        """Get the currently applied status effect.

        Returns:
            The currently applied status effect.
        """
        return self._applied_status_effect

    def upgrade(self: EntityAttributeBase, increase: Callable[[int], float]) -> bool:
        """Upgrade the entity attribute to the next level if possible.

        Args:
            increase: The exponential lambda function which calculates the next level's
                value based on the current level.

        Returns:
            Whether the attribute upgrade was successful or not.

        Raises:
            EntityAttributeError: The entity attribute cannot be upgraded.
        """
        # Check if the attribute can be upgraded
        if not self.upgradable:
            raise EntityAttributeError(
                name=self.__class__.__name__,
                error="be upgraded",
            )

        # Check if the current level is below the level limit
        if self.current_level >= self.level_limit:
            return False

        # Upgrade the attribute based on the difference between the current level and
        # the next
        diff = increase(self.current_level + 1) - increase(self.current_level)
        self._max_value += diff
        self._current_level += 1
        self.value += diff
        return True

    def apply_instant_effect(
        self: EntityAttributeBase,
        increase: Callable[[int], float],
        level: int,
    ) -> bool:
        """Apply an instant effect to the entity attribute if possible.

        Args:
            increase: The exponential lambda function which calculates the next level's
                value based on the current level.
            level: The level to initialise the instant effect at.

        Returns:
            Whether the instant effect could be applied or not.

        Raises:
            EntityAttributeError: The entity attribute cannot have an instant effect.
        """
        # Check if the attribute can have an instant effect
        if not self.instant_effect:
            raise EntityAttributeError(
                name=self.__class__.__name__,
                error="have an instant effect",
            )

        # Check if the attribute's value is already at max
        if self.value == self.max_value:
            return False

        # Add the instant effect to the attribute
        self.value += increase(level)
        return True

    def apply_status_effect(
        self: EntityAttributeBase,
        increase: Callable[[int], float],
        duration: Callable[[int], float],
        level: int,
    ) -> bool:
        """Apply a status effect to the attribute if possible.

        Args:
            increase: The exponential lambda function which calculates the next level's
                value based on the current level.
            duration: The exponential lambda function which calculates the next level's
                duration based on the current level.
            level: The level to initialise the status effect at.

        Returns:
            Whether the status effect could be applied or not.

        Raises:
            EntityAttributeError: The entity attribute cannot have a status effect.
        """
        # Check if the attribute can have a status effect
        if not self.status_effect:
            raise EntityAttributeError(
                name=self.__class__.__name__,
                error="have a status effect",
            )

        # Check if the attribute already has a status effect applied
        if self.applied_status_effect:
            return False

        # Apply the status effect to this attribute
        self._applied_status_effect = StatusEffect(
            increase(level),
            duration(level),
            self.value,
            self.max_value,
        )
        self._value += self._applied_status_effect.value
        self._max_value += self._applied_status_effect.value
        return True

    def update_status_effect(self: EntityAttributeBase, delta_time: float) -> bool:
        """Update the currently applied status effect.

        Args:
            delta_time: Time interval since the last time the function was called.

        Returns:
            Whether the status effect update was successful or not.
        """
        # Check if there isn't a status effect already applied
        if not self.applied_status_effect:
            return False

        # Update the time counter and check if the status effect should be removed
        self.applied_status_effect.time_counter += delta_time
        if (
            self.applied_status_effect.time_counter
            >= self.applied_status_effect.duration
        ):
            self.value = min(self.value, self.applied_status_effect.original_value)
            self._max_value = self.applied_status_effect.original_max_value
            self._applied_status_effect = None
        return True

    def __repr__(self: EntityAttributeBase) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<{self.__class__.__name__} (Value={self.value}) (Max"
            f" value={self.max_value}) (Level={self.current_level}/{self.level_limit})>"
        )


class Armour(EntityAttributeBase):
    """Allows a game object to have an armour attribute."""

    # Class variables
    component_type: ComponentType = ComponentType.ARMOUR


class ArmourRegenCooldown(EntityAttributeBase):
    """Allows a game object to have an armour regen cooldown attribute."""

    # Class variables
    component_type: ComponentType = ComponentType.ARMOUR_REGEN_COOLDOWN
    instant_effect: bool = False
    maximum: bool = False


class FireRatePenalty(EntityAttributeBase):
    """Allows a game object to have a fire rate penalty attribute."""

    # Class variables
    component_type: ComponentType = ComponentType.FIRE_RATE_PENALTY
    instant_effect: bool = False
    maximum: bool = False


class Health(EntityAttributeBase):
    """Allows a game object to have a health attribute."""

    # Class variables
    component_type: ComponentType = ComponentType.HEALTH


class Money(EntityAttributeBase):
    """Allows a game object to have a money attribute."""

    # Class variables
    component_type: ComponentType = ComponentType.MONEY
    instant_effect: bool = False
    maximum: bool = False
    status_effect: bool = False
    upgradable: bool = False


class MovementForce(EntityAttributeBase):
    """Allows a game object to have a movement force attribute."""

    # Class variables
    component_type: ComponentType = ComponentType.MOVEMENT_FORCE
    instant_effect: bool = False
    maximum: bool = False


class ViewDistance(EntityAttributeBase):
    """Allows a game object to have a view distance attribute."""

    # Class variables
    component_type: ComponentType = ComponentType.VIEW_DISTANCE
    instant_effect: bool = False
    maximum: bool = False
