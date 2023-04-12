"""Manages the different enums related to the game objects."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import NamedTuple


__all__ = (
    "ComponentType",
    "ActionableData",
    "AreaOfEffectAttackData",
    "AttackerData",
    "CollectibleData",
    "InventoryData",
    "MeleeAttackData",
    "RangedAttackData",
)


class ComponentType(Enum):
    """Stores the different types of components available."""

    ACTIONABLE = auto()
    AREA_OF_EFFECT_ATTACK = auto()
    ATTACKER = auto()
    COLLECTIBLE = auto()
    INVENTORY = auto()
    MELEE_ATTACK = auto()
    RANGED_ATTACK = auto()


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
    component_data: dict[ComponentType, type[ComponentData]]

# TODO: Textures, armour_regen, level_limit, view_distance, entity attributes, player
#  upgrades, instant effects, status effects
# TODO: Should try and redo textures script and not sure about characteristic base in
#  draw.io diagram
# TODO: Maybe just have the constructor data be in the class instead of trying to
#  dynamically load it
