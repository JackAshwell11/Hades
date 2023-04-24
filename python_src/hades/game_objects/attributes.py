"""Manages the different game object attributes available."""
from __future__ import annotations

# Custom
from hades.game_objects.base import ComponentType

__all__ = ()


class EntityAttributeBase:
    """The base class for all entity attributes."""

    # Class variables
    maximum: bool = True
    variable: bool = True
    status_effect: bool = True

    def __init__(self: EntityAttributeBase) -> None:
        """Initialise the object."""
        self._value: float = 0

    @property
    def value(self: EntityAttributeBase) -> float:
        """Get the entity attribute's value.

        Returns
        -------
        float
            The entity attribute's value.
        """
        return self._value

    @value.setter
    def value(self: EntityAttributeBase, new_value: float) -> None:
        """Set the entity attribute's value.

        Parameters
        ----------
        new_value: float
            The new entity attribute's value.
        """
        self._value = new_value


class Armour(EntityAttributeBase):
    """Allows a game object to have an armour attribute."""


class ArmourRegenCooldown(EntityAttributeBase):
    """Allows a game object to have an armour regen cooldown attribute."""

    # Class variables
    variable: bool = False


class FireRatePenalty(EntityAttributeBase):
    """Allows a game object to have a fire rate penalty attribute."""

    # Class variables
    variable: bool = False


class Health(EntityAttributeBase):
    """Allows a game object to have a health attribute."""


class Money(EntityAttributeBase):
    """Allows a game object to have a money attribute."""

    # Class variables
    maximum: bool = False
    status_effect: bool = False


class SpeedMultiplier(EntityAttributeBase):
    """Allows a game object to have a speed multiplier attribute."""

    # Class variables
    variable: bool = False

    @EntityAttributeBase.value.setter
    def value(self: SpeedMultiplier, new_value: float) -> None:
        """Set the entity attribute's value.

        Parameters
        ----------
        new_value: float
            The new entity attribute's value.
        """
        self._value = new_value


class ViewDistance(EntityAttributeBase):
    """Allows a game object to have a view distance attribute."""

    # Class variables
    variable: bool = False


class Attributes:
    """Allows a game object to have various attributes that can affect its behaviour."""

    __slots__ = (
        "armour",
        "armour_regen_cooldown",
        "fire_rate_penalty",
        "health",
        "money",
        "speed_multiplier",
        "view_distance",
    )

    # Class variables
    component_type: ComponentType = ComponentType.ATTRIBUTES

    def __init__(
        self: Attributes,
        *,
        armour: Armour | None = None,
        armour_regen_cooldown: ArmourRegenCooldown | None = None,
        fire_rate_penalty: FireRatePenalty | None = None,
        health: Health | None = None,
        money: Money | None = None,
        speed_multiplier: SpeedMultiplier | None = None,
        view_distance: ViewDistance | None = None,
    ) -> None:
        """Initialise the object.

        Parameters
        ----------
        armour: Armour | None
            The optional armour attribute.
        armour_regen_cooldown: ArmourRegenCooldown | None
            The optional armour regen cooldown attribute.
        fire_rate_penalty: FireRatePenalty | None
            The optional fire rate penalty attribute.
        health: Health | None
            The optional health attribute.
        money: Money | None
            The optional money attribute.
        speed_multiplier: SpeedMultiplier | None
            The optional speed multiplier attribute.
        view_distance: ViewDistance | None
            The optional view distance attribute.
        """
        self.armour: Armour | None = armour
        self.armour_regen_cooldown: ArmourRegenCooldown | None = armour_regen_cooldown
        self.fire_rate_penalty: FireRatePenalty | None = fire_rate_penalty
        self.health: Health | None = health
        self.money: Money | None = money
        self.speed_multiplier: SpeedMultiplier | None = speed_multiplier
        self.view_distance: ViewDistance | None = view_distance


# TODO: Entity attribute specific attributes to implement:
#       player upgrades
#       instant effects
#       status effects
#       level limit
