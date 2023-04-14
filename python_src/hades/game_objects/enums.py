"""Manages the different enums related to the game objects."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = (
    "ActionableData",
    "AreaOfEffectAttackData",
    "AttackerData",
    "CollectibleData",
    "ComponentType",
    "EntityAttributeData",
    "GameObjectData",
    "InventoryData",
    "MeleeAttackData",
    "RangedAttackData",
)


class ComponentType(Enum):
    """Stores the different types of components available."""

    ACTIONABLE = auto()
    AREA_OF_EFFECT_ATTACK = auto()
    ARMOUR = auto()
    ARMOUR_REGEN = auto()
    ATTACKER = auto()
    COLLECTIBLE = auto()
    FIRE_RATE_PENALTY = auto()
    HEALTH = auto()
    INVENTORY = auto()
    MELEE_ATTACK = auto()
    MONEY = auto()
    RANGED_ATTACK = auto()
    SPEED_MULTIPLIER = auto()


class ComponentData(NamedTuple):
    """The base class for data about a component."""


class ActionableData(ComponentData):
    """Stores data about the actionable component.

    item_text: str
        The text to display when the player is near this collectible.
    """


class AreaOfEffectAttackData(ComponentData):
    """Stores data about the area of effect attack component.

    damage: int
        The damage the game object deals.
    attack_cooldown: int
        The time duration between attacks.
    attack_range: int
        The range of the game object's attack.
    """

    damage: int
    attack_cooldown: int
    attack_range: int


class AttackerData(ComponentData):
    """Stores data about the attacker component."""


class CollectibleData(ComponentData):
    """Stores data about the collectible component.

    item_text: str
        The text to display when the player is near this collectible.
    """

    item_text: str = "Press E to pick up"


class EntityAttributeData(ComponentData):
    """Stores data about the entity attribute components.

    increase: Callable[[int], float]
        The exponential lambda function which calculates the next level's value based on
        the current level.
    maximum: bool
        Whether this attribute has a maximum level or not.
    status_effect: bool
        Whether this attribute can have a status effect applied to it or not.
    variable: bool
        Whether this attribute can change from it current value or not.
    """

    increase: Callable[[int], float]
    maximum: bool = True
    status_effect: bool = False
    variable: bool = False


class InventoryData(ComponentData):
    """Stores data about the inventory component."""

    width: int
    height: int


class MeleeAttackData(ComponentData):
    """Stores data about the melee attack component.

    damage: int
        The damage the game object deals.
    attack_cooldown: int
        The time duration between attacks.
    attack_range: int
        The range of the game object's attack.
    """

    damage: int
    attack_cooldown: int
    attack_range: int


class RangedAttackData(ComponentData):
    """Stores data about the ranged attack component.

    damage: int
        The damage the game object deals.
    attack_cooldown: int
        The time duration between attacks.
    max_bullet_range: int
        The max range of the bullet.
    """

    damage: int
    attack_cooldown: int
    max_bullet_range: int


class GameObjectData(NamedTuple):
    """The base class for storing general data about a game object.

    name: str
        The name of the game object.
    component_data: dict[ComponentType, type[ComponentData]]
        The components that are available to this game object.
    """

    name: str
    textures: list
    component_data: dict[ComponentType, type[ComponentData]]


# TODO: armour_regen, level_limit, view_distance, player upgrades, instant effects, status effects
# TODO: Should try and redo textures script
