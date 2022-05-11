from __future__ import annotations

# Builtin
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# Entity attribute upgrades
class UpgradeAttribute(Enum):
    """Stores the types of attributes for the entity which can be upgraded."""

    HEALTH = "health"
    ARMOUR = "armour"
    SPEED = "speed"
    REGEN_COOLDOWN = "regen cooldown"
    MELEE_ATTACK = "melee attack"
    AREA_OF_EFFECT_ATTACK = "area of effect attack"
    POTION_DURATION = "potion duration"
    RANGED_ATTACK = "ranged attack"


# Entity upgrade sections
class UpgradeSection(Enum):
    """Stores the sections that can be upgraded by the player improving various
    attributes."""

    ENDURANCE = "endurance"
    DEFENCE = "defence"
    STRENGTH = "strength"
    INTELLIGENCE = "intelligence"


@dataclass
class EntityUpgradeData:
    """
    Stores an upgrade that is available to the entity. If the cost function is set to
    -1, then the upgrade does not exist for the entity.

    section_type: UpgradeSection
        The type of upgrade this instance represents.
    cost: Callable[[int], float]
        The exponential lambda function which calculates the next level's cost based on
        the current level.
    level_limit: int
        The maximum level this upgrade can go to.
    upgrades: list[AttributeUpgrade]
        The list of attribute upgrades which are included in this instance.
    """

    section_type: UpgradeSection = field(kw_only=True)
    cost: Callable[[int], float] = field(kw_only=True)
    level_limit: int = field(kw_only=True)
    upgrades: list[AttributeUpgrade] = field(kw_only=True)


@dataclass
class AttributeUpgrade:
    """
    Stores an attribute upgrade that is available to the entity.

    attribute_type: UpgradeAttribute
        The type of attribute which this upgrade targets.
    increase: Callable[[int], float]
        The exponential lambda function which calculates the next level's value based on
        the current level.
    """

    attribute_type: UpgradeAttribute = field(kw_only=True)
    increase: Callable[[int], float] = field(kw_only=True)


PLAYER_LEVELS = [
    EntityUpgradeData(
        section_type=UpgradeSection.ENDURANCE,
        cost=lambda current_level: 1 * 3**current_level,
        level_limit=5,
        upgrades=[
            AttributeUpgrade(
                attribute_type=UpgradeAttribute.HEALTH,
                increase=lambda current_level: 100 * 1.4**current_level,
            ),
            AttributeUpgrade(
                attribute_type=UpgradeAttribute.SPEED,
                increase=lambda current_level: 200 * 1.4**current_level,
            ),
        ],
    ),
    EntityUpgradeData(
        section_type=UpgradeSection.DEFENCE,
        cost=lambda current_level: 1 * 3**current_level,
        level_limit=5,
        upgrades=[
            AttributeUpgrade(
                attribute_type=UpgradeAttribute.ARMOUR,
                increase=lambda current_level: 20 * 1.4**current_level,
            ),
            AttributeUpgrade(
                attribute_type=UpgradeAttribute.REGEN_COOLDOWN,
                increase=lambda current_level: 2 * 0.5**current_level,
            ),
        ],
    ),
]

# Enemy characters
ENEMY1_LEVELS = [
    EntityUpgradeData(
        section_type=UpgradeSection.ENDURANCE,
        cost=lambda current_level: -1,
        level_limit=5,
        upgrades=[
            AttributeUpgrade(
                attribute_type=UpgradeAttribute.HEALTH,
                increase=lambda current_level: 10 * 1.4**current_level,
            ),
            AttributeUpgrade(
                attribute_type=UpgradeAttribute.SPEED,
                increase=lambda current_level: 50 * 1.4**current_level,
            ),
        ],
    ),
    EntityUpgradeData(
        section_type=UpgradeSection.DEFENCE,
        cost=lambda current_level: -1,
        level_limit=5,
        upgrades=[
            AttributeUpgrade(
                attribute_type=UpgradeAttribute.ARMOUR,
                increase=lambda current_level: 10 * 1.4**current_level,
            ),
            AttributeUpgrade(
                attribute_type=UpgradeAttribute.REGEN_COOLDOWN,
                increase=lambda current_level: 3 * 0.6**current_level,
            ),
        ],
    ),
]
