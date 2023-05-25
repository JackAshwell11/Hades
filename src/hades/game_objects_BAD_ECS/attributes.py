"""Manages the different attributes available."""
from __future__ import annotations

__all__ = (
    "ArmourMixin",
    "ArmourRegenCooldownMixin",
    "FireRatePenaltyMixin",
    "HealthMixin",
    "MoneyMixin",
    "SpeedMultiplierMixin",
)


class HealthMixin:
    """Allows a game object to have health."""

    # Class variables
    __maximum: bool = True
    __status_effect: bool = True
    __variable: bool = True

    def __init__(self: HealthMixin, **kwargs) -> None:
        """Initialise the object.

        Parameters
        ----------
        kwargs: TODO
            The keyword arguments for the health attribute.
        """
        super().__init__(**kwargs)
        self._health: int = 0

    @property
    def health(self: HealthMixin) -> int:
        """Get the game object's health attribute.

        Returns
        -------
        int
            The game object's health attribute.
        """
        return self._health

    @health.setter
    def health(self: HealthMixin, value: int) -> None:
        """Set the game object's health attribute.

        Parameters
        ----------
        value: int
            The game object's new health value.
        """
        # Check if the attribute can be set
        if not self.__variable:
            raise ValueError("This attribute's value cannot be set.")

        # Update the attribute value with the new value
        self._health = value

        # Check if the attribute value exceeds the max. If so, set it to the max
        if self.health > HEALTH_MAX_VALUE:
            self._health = HEALTH_MAX_VALUE


class ArmourMixin:
    """Allows a game object to have armour."""

    # Class variables
    __maximum: bool = True
    __status_effect: bool = True
    __variable: bool = True

    def __init__(self: ArmourMixin, **kwargs) -> None:
        """Initialise the object.

        Parameters
        ----------
        kwargs: TODO
            The keyword arguments for the armour attribute.
        """
        super().__init__(**kwargs)
        self._armour: int = 0
        print("armour", kwargs, self.__variable)

    @property
    def armour(self: ArmourMixin) -> int:
        """Get the game object's armour attribute.

        Returns
        -------
        int
            The game object's armour attribute.
        """
        return self._armour

    @armour.setter
    def armour(self: ArmourMixin, value: int) -> None:
        """Set the game object's armour attribute.

        Parameters
        ----------
        value: int
            The game object's new armour value.
        """
        # Check if the attribute can be set
        if not self.__variable:
            raise ValueError("This attribute's value cannot be set.")

        # Update the attribute value with the new value
        self._armour = value

        # Check if the attribute value exceeds the max. If so, set it to the max
        if self.armour > ARMOUR_MAX_VALUE:
            self._armour = ARMOUR_MAX_VALUE


class SpeedMultiplierMixin:
    """Allows a game object to have a speed multiplier."""

    # Class variables
    __maximum: bool = True
    __status_effect: bool = True
    __variable: bool = False

    def __init__(self: SpeedMultiplierMixin, **kwargs) -> None:
        """Initialise the object.

        Parameters
        ----------
        kwargs: TODO
            The keyword arguments for the speed multiplier attribute.
        """
        super().__init__(**kwargs)
        self._speed_multiplier: float = 0
        print("speed multiplier", kwargs, self.__variable)

    @property
    def speed_multiplier(self: SpeedMultiplierMixin) -> float:
        """Get the game object's speed multiplier attribute.

        Returns
        -------
        float
            The game object's speed multiplier attribute.
        """
        return self._speed_multiplier

    @speed_multiplier.setter
    def speed_multiplier(self: SpeedMultiplierMixin, value: float) -> None:
        """Set the game object's speed multiplier attribute.

        Parameters
        ----------
        value: int
            The game object's new speed multiplier value.
        """
        # Check if the attribute can be set
        if not self.__variable:
            raise ValueError("This attribute's value cannot be set.")

        # Update the attribute value with the new value
        self._speed_multiplier = value

        # Check if the attribute value exceeds the max. If so, set it to the max
        if self.speed_multiplier > SPEED_MULTIPLIER_MAX_VALUE:
            self._speed_multiplier = SPEED_MULTIPLIER_MAX_VALUE


class ArmourRegenCooldownMixin:
    """Allows a game object to have an armour regen cooldown.

    Attributes
    ----------
    armour_regen_cooldown: float
        The game object's armour regen cooldown attribute.
    """

    # Class variables
    __maximum: bool = True
    __status_effect: bool = True
    __variable: bool = False

    def __init__(self: ArmourRegenCooldownMixin, **kwargs) -> None:
        """Initialise the object.

        Parameters
        ----------
        kwargs: TODO
            The keyword arguments for the armour regen attribute.
        """
        super().__init__(**kwargs)
        self.armour_regen_cooldown: float = 0
        print("armour regen cooldown", kwargs, self.__variable)


class FireRatePenaltyMixin:
    """Allows a game object to have a fire rate penalty.

    Attributes
    ----------
    fire_rate_penalty: float
        The game object's fire rate penalty attribute.
    """

    # Class variables
    __maximum: bool = True
    __status_effect: bool = True
    __variable: bool = False

    def __init__(self: FireRatePenaltyMixin, **kwargs) -> None:
        """Initialise the object.

        Parameters
        ----------
        kwargs: TODO
            The keyword arguments for the fire rate penalty attribute.
        """
        super().__init__(**kwargs)
        self.fire_rate_penalty: float = 0
        print("fire rate penalty", kwargs, self.__variable)


class MoneyMixin:
    """Allows a game object to have money.

    Attributes
    ----------
    money: int
        The game object's money attribute.
    """

    # Class variables
    __maximum: bool = True
    __status_effect: bool = True
    __variable: bool = True

    def __init__(self: MoneyMixin, **kwargs) -> None:
        """Initialise the object.

        Parameters
        ----------
        kwargs: TODO
            The keyword arguments for the money attribute.
        """
        super().__init__(**kwargs)
        self.money: int = 0
        print("money", kwargs, self.__variable)
