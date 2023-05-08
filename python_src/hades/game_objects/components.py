"""Manages various components available to the game objects."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Generic, TypeVar

# Pip
import arcade

# Custom
from hades.constants import SPRITE_SCALE, GameObjectType
from hades.game_objects.base import ComponentType, GameObjectComponent

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.base import ComponentData
    from hades.textures import TextureType

__all__ = (
    "HadesSprite",
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

    def __init__(self: InstantEffects, component_data: ComponentData) -> None:
        """Initialise the object.

        Args:
            component_data: The data for the components.
        """
        super().__init__(component_data)
        level_limit, instant_effects = component_data["instant_effects"]
        self.instant_effects: dict[ComponentType, Callable[[int], float]] = (
            instant_effects
        )
        self.level_limit: int = level_limit

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

    def __init__(self: Inventory[T], component_data: ComponentData) -> None:
        """Initialise the object.

        Args:
            component_data: The data for the components.
        """
        super().__init__(component_data)
        self.width: int = component_data["inventory_width"]
        self.height: int = component_data["inventory_height"]
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

    def __init__(self: StatusEffects, component_data: ComponentData) -> None:
        """Initialise the object.

        Args:
            component_data: The data for the components.
        """
        super().__init__(component_data)
        level_limit, status_effects = component_data["status_effects"]
        self.status_effects: dict[
            ComponentType,
            tuple[Callable[[int], float], Callable[[int], float]],
        ] = status_effects
        self.level_limit: int = level_limit

    def __repr__(self: StatusEffects) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<StatusEffects (Level limit={self.level_limit})>"


class HadesSprite(arcade.Sprite, GameObjectComponent):
    """Represents a game object in the game.

    Attributes:
        textures_dict: The textures which represent this game object.
    """

    def __init__(
        self: HadesSprite,
        game_object_type: GameObjectType,
        position: tuple[int, int],
        component_data: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_type: The type of the game object.
            position: The position of the game object on the screen.
            component_data: The data for the components.
        """
        super().__init__(scale=SPRITE_SCALE)
        self.game_object_type: GameObjectType = game_object_type
        self.position = position
        self.textures_dict: dict[TextureType, arcade.Texture] = {
            texture: texture.value  # type: ignore[misc]
            for texture in component_data["texture_types"]
        }
        self.blocking: bool = component_data["blocking"]

    def __repr__(self: HadesSprite) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<HadesSprite (Game object type={self.game_object_type}) (Texture"
            f" count={len(self.textures_dict)}) (Blocking={self.blocking})>"
        )
