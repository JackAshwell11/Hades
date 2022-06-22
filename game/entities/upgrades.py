"""
Manages upgrading of the player's attributes.
"""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Custom
from game.constants.entity import UpgradeAttribute

if TYPE_CHECKING:
    from game.constants.entity import AttributeUpgradeData, EntityUpgradeData
    from game.entities.player import Player
    from game.views.shop_view import SectionUpgradeButton

# Get the logger
logger = logging.getLogger(__name__)


class UpgradableAttributeBase:
    """
    The base class for all player attributes that can be upgraded.

    Parameters
    ----------
    parent_section: UpgradableSection
        The reference to the parent upgradable section object.
    player: Player
        The reference to the player object.
    attribute_upgrade_data: AttributeUpgradeData
        The upgrade data for this attribute.
    """

    __slots__ = (
        "parent_section",
        "player",
        "attribute_upgrade_data",
    )

    def __init__(
        self,
        parent_section: UpgradableSection,
        player: Player,
        attribute_upgrade_data: AttributeUpgradeData,
    ) -> None:
        self.parent_section: UpgradableSection = parent_section
        self.player: Player = player
        self.attribute_upgrade_data: AttributeUpgradeData = attribute_upgrade_data

    def __repr__(self) -> str:
        return f"<UpgradableAttribute (Player={self.player})>"

    def upgrade_attribute(self) -> None:
        """
        Upgrades the corresponding player attribute which matches this class.

        Raises
        ------
        NotImplementedError
            The function is not implemented.
        """
        raise NotImplementedError


class HealthUpgradableAttribute(UpgradableAttributeBase):
    """Manages upgrading of the player's health attribute."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<HealthUpgradableAttribute (Player={self.player})>"

    def upgrade_attribute(self) -> None:
        """Upgrades the player's health attribute."""
        # Find the difference between the current level and the next level and increase
        # the player's health by that difference
        diff = self.attribute_upgrade_data.increase(
            self.parent_section.current_level
        ) - self.attribute_upgrade_data.increase(self.parent_section.current_level - 1)
        self.player.health += diff
        self.player.max_health += diff


class ArmourUpgradableAttribute(UpgradableAttributeBase):
    """Manages upgrading of the player's armour attribute."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<ArmourUpgradableAttribute (Player={self.player})>"

    def upgrade_attribute(self) -> None:
        """Upgrades the player's armour attribute."""
        # Find the difference between the current level and the next level and increase
        # the player's armour by that difference
        diff = self.attribute_upgrade_data.increase(
            self.parent_section.current_level
        ) - self.attribute_upgrade_data.increase(self.parent_section.current_level - 1)
        self.player.armour += diff
        self.player.max_armour += diff


class SpeedUpgradableAttribute(UpgradableAttributeBase):
    """Manages upgrading of the player's speed attribute."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<SpeedUpgradableAttribute (Player={self.player})>"

    def upgrade_attribute(self) -> None:
        """Upgrades the player's speed attribute."""
        # Find the difference between the current level and the next level and increase
        # the player's speed by that difference
        diff = self.attribute_upgrade_data.increase(
            self.parent_section.current_level
        ) - self.attribute_upgrade_data.increase(self.parent_section.current_level - 1)
        self.player.max_velocity += diff


class RegenCooldownUpgradableAttribute(UpgradableAttributeBase):
    """Manages upgrading of the player's regen cooldown attribute."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<RegenCooldownUpgradableAttribute (Player={self.player})>"

    def upgrade_attribute(self) -> None:
        """Upgrades the player's regen cooldown attribute."""
        # Find the difference between the current level and the next level and increase
        # the player's regen cooldown by that difference
        diff = self.attribute_upgrade_data.increase(
            self.parent_section.current_level
        ) - self.attribute_upgrade_data.increase(self.parent_section.current_level - 1)
        self.player.armour_regen_cooldown += diff


UPGRADABLE_ATTRIBUTES = {
    UpgradeAttribute.HEALTH: HealthUpgradableAttribute,
    UpgradeAttribute.ARMOUR: ArmourUpgradableAttribute,
    UpgradeAttribute.SPEED: SpeedUpgradableAttribute,
    UpgradeAttribute.REGEN_COOLDOWN: RegenCooldownUpgradableAttribute,
}


def create_attribute_upgrade(
    upgrade_attribute_type: UpgradeAttribute,
    parent_section: UpgradableSection,
    player: Player,
    attribute_upgrade_data: AttributeUpgradeData,
) -> UpgradableAttributeBase:
    """
    Determines which upgradable attribute class should be initialised based on a given
    upgrade attribute type.

    Parameters
    ----------
    upgrade_attribute_type: UpgradeAttribute
        The upgradable attribute to create.
    parent_section: UpgradableSection
        The reference to the parent upgradable section object.
    player: Player
        The reference to the player object.
    attribute_upgrade_data: AttributeUpgradeData
        The upgrade data for this attribute.
    """
    # Get the upgradable attribute class type which manages the given upgradable
    # attribute
    cls = UPGRADABLE_ATTRIBUTES[upgrade_attribute_type]
    logger.debug(
        "Selected upgradable attribute %r for upgrade type %r",
        cls,
        upgrade_attribute_type,
    )

    # Initialise the class with the given parameters
    return cls(parent_section, player, attribute_upgrade_data)


class UpgradableSection:
    """
    Represents a player section that can be upgraded.

    Parameters
    ----------
    player: Player
        The reference to the player object.
    entity_upgrade_data: EntityUpgradeData
        The upgrade data for this section.
    current_level: int
        The current level of this section.
    """

    __slots__ = (
        "player",
        "entity_upgrade_data",
        "current_level",
        "attributes",
    )

    def __init__(
        self, player: Player, entity_upgrade_data: EntityUpgradeData, current_level: int
    ) -> None:
        self.player: Player = player
        self.entity_upgrade_data: EntityUpgradeData = entity_upgrade_data
        self.current_level: int = current_level
        self.attributes = [
            create_attribute_upgrade(
                attribute_upgrade.attribute_type, self, player, attribute_upgrade
            )
            for attribute_upgrade in entity_upgrade_data.upgrades
        ]

    def __repr__(self) -> str:
        return (
            f"<UpgradableSection (Player={self.player}) (Current"
            f" level={self.current_level}) (Level limit={self.level_limit})>"
        )

    @property
    def next_level_cost(self) -> int:
        """
        Gets the cost for the next level.

        Returns
        -------
        int
            The next level cost.
        """
        return round(self.entity_upgrade_data.cost(self.current_level))

    @property
    def level_limit(self) -> int:
        """
        Gets the maximum level for the player's upgrades.

        Returns
        -------
        int
            The maximum level for the player's upgrades.
        """
        return self.player.entity_data.upgrade_level_limit

    def upgrade_section(self, shop_button: SectionUpgradeButton) -> None:
        """
        Upgrades the player section if the player has enough money.

        Parameters
        ----------
        shop_button: SectionUpgradeButton
            The shop section upgrade button which called this function.
        """
        # Check if the player has enough money
        if (
            self.player.money >= self.next_level_cost
            and self.current_level < self.level_limit
        ):
            # Subtract the cost from the player's money and upgrade each attribute this
            # section manages
            logger.debug("Upgrading section %r", self.entity_upgrade_data.section_type)
            self.player.money -= self.next_level_cost
            for attribute_upgrade in self.attributes:
                logger.debug("Upgrading attribute %r", attribute_upgrade)
                attribute_upgrade.upgrade_attribute()

            # Increase this section's level
            self.current_level += 1

            # Update the shop button text
            shop_button.text = (
                f"{self.entity_upgrade_data.section_type.value} -"
                f" {self.next_level_cost}"
            )
