"""Manages various components available to the game objects."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Generic, TypeVar

from hades.constants import ComponentType

# Custom
from hades.game_objects.base import GameObjectComponent

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentData
    from hades.game_objects.system import ECS

__all__ = (
    "InstantEffects",
    "Inventory",
    "InventorySpaceError",
    "StatusEffects",
)

# Define a generic type for the inventory
T = TypeVar("T")


class InventorySpaceError(Exception):
    """Raised when there is a space problem with the inventory."""

    def __init__(self: InventorySpaceError, *, full: bool) -> None:
        """Initialise the object.

        Args:
            full: Whether the inventory is empty or full.
        """
        super().__init__(f"The inventory is {'full' if full else 'empty'}.")


class InstantEffects(GameObjectComponent):
    """Allows a game object to provide instant effects."""

    __slots__ = ("instant_effects", "level_limit")

    # Class variables
    component_type: ComponentType = ComponentType.INSTANT_EFFECTS

    def __init__(
        self: InstantEffects,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.level_limit, self.instant_effects = component_data["instant_effects"]

    def __repr__(self: InstantEffects) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<InstantEffects (Level limit={self.level_limit})>"


class Inventory(Generic[T], GameObjectComponent):
    """Allows a game object to have a fixed size inventory.

    Attributes:
        inventory: The game object's inventory.
    """

    __slots__ = (
        "width",
        "height",
        "inventory",
    )

    # Class variables
    component_type: ComponentType = ComponentType.INVENTORY

    def __init__(
        self: Inventory[T],
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.width, self.height = component_data["inventory_size"]
        self.inventory: list[T] = []

    def add_item_to_inventory(self: Inventory[T], item: T) -> None:
        """Add an item to the inventory.

        Args:
            item: The item to add to the inventory.

        Raises:
            InventorySpaceError: The inventory is full.
        """
        if len(self.inventory) == self.width * self.height:
            raise InventorySpaceError(full=True)
        self.inventory.append(item)

    def remove_item_from_inventory(self: Inventory[T], index: int) -> T:
        """Remove an item at a specific index.

        Args:
            index: The index to remove an item at.

        Returns:
            The item at position `index` in the inventory.

        Raises:
            InventorySpaceError: The inventory is empty.
        """
        if len(self.inventory) < index:
            raise InventorySpaceError(full=False)
        return self.inventory.pop(index)

    def __repr__(self: Inventory[T]) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<Inventory (Width={self.width}) (Height={self.height})>"


class StatusEffects(GameObjectComponent):
    """Allows a game object to provide status effects."""

    __slots__ = ("status_effects", "level_limit")

    # Class variables
    component_type: ComponentType = ComponentType.STATUS_EFFECTS

    def __init__(
        self: StatusEffects,
        game_object_id: int,
        system: ECS,
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
            component_data: The data for the components.
        """
        super().__init__(game_object_id, system, component_data)
        self.level_limit, self.status_effects = component_data["status_effects"]

    def __repr__(self: StatusEffects) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<StatusEffects (Level limit={self.level_limit})>"
