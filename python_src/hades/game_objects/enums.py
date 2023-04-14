"""Manages the different enums related to the game objects."""
from __future__ import annotations

# Builtin
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

# Custom
from hades.game_objects.attacks import Attacker, RangedAttackMixin, MeleeAttackMixin, AreaOfEffectAttackMixin
from hades.game_objects.attributes import Health, Armour
from hades.game_objects.components import ActionableMixin, CollectibleMixin, Inventory

if TYPE_CHECKING:
    from arcade import Texture

    from hades.textures import MovingTextureType, NonMovingTextureType

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

    ACTIONABLE = ActionableMixin
    AREA_OF_EFFECT_ATTACK = AreaOfEffectAttackMixin
    ARMOUR = Armour
    ARMOUR_REGEN = auto()
    ATTACKER = Attacker
    COLLECTIBLE = CollectibleMixin
    FIRE_RATE_PENALTY = auto()
    HEALTH = Health
    INVENTORY = Inventory
    MELEE_ATTACK = MeleeAttackMixin
    MONEY = auto()
    RANGED_ATTACK = RangedAttackMixin
    SPEED_MULTIPLIER = auto()


# class InstantEffectType(Enum):
#     """Stores the type of instant effects that can be applied to an entity."""
#
#
#
# class StatusEffectType(Enum):
#     """Stores the type of status effects that can be applied to an entity."""
#


class ActionableData(NamedTuple):
    """Stores data about the actionable component.

    item_text: str
        The text to display when the player is near this collectible.
    """


class AreaOfEffectAttackData(NamedTuple):
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


class AttackerData(NamedTuple):
    """Stores data about the attacker component."""


class CollectibleData(NamedTuple):
    """Stores data about the collectible component.

    item_text: str
        The text to display when the player is near this collectible.
    """

    item_text: str = "Press E to pick up"


class EntityAttributeData(NamedTuple):
    """Stores data about the entity attribute components.

    value: float
        The attribute's initial value.
    maximum: bool
        Whether this attribute has a maximum level or not.
    variable: bool
        Whether this attribute can change from it current value or not.
    """

    value: float
    maximum: bool
    variable: bool


class InventoryData(NamedTuple):
    """Stores data about the inventory component."""

    width: int
    height: int


class MeleeAttackData(NamedTuple):
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


class RangedAttackData(NamedTuple):
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
    textures: dict[MovingTextureType |NonMovingTextureType, Texture | list[Texture]]
        The moving or non-moving textures associated with this game object.
    component_data: dict[ComponentType, type[ComponentData]]
        The components that are available to this game object.
    static: bool
        Whether the game object is allowed to move or not.
    """

    name: str
    textures: dict[MovingTextureType, NonMovingTextureType, Texture | list[Texture]]
    component_data: dict[ComponentType, ActionableData | AreaOfEffectAttackData | AttackerData | CollectibleData | EntityAttributeData | InventoryData | MeleeAttackData | RangedAttackData]
    static: bool = False


# TODO: armour_regen, level_limit, view_distance, player upgrades, instant effects, status effects
