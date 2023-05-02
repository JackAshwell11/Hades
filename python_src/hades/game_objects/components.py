"""Manages various components available to the game objects."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, Generic, TypeVar

# Pip
import arcade

# Custom
from hades.constants import SPRITE_SCALE
from hades.game_objects.base import ComponentType, GameObjectComponent

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.game_objects.base import D
    from hades.textures import TextureType

__all__ = (
    "Graphics",
    "InstantEffect",
    "Inventory",
    "InventorySpaceError",
    "StatusEffect",
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


class Graphics(arcade.Sprite, GameObjectComponent):
    """Allows a game object to be drawn on the screen and interact with Arcade.

    Attributes:
        textures_dict: The textures which represent this game object.
    """

    # Class variables
    component_type: ComponentType = ComponentType.GRAPHICS

    def __init__(
        self: Graphics,
        texture_types: set[TextureType],
        *,
        blocking: bool = False,
        **_: D,
    ) -> None:
        """Initialise the object.

        Args:
            texture_types: The textures that relate to this component.
            blocking: Whether this component is blocking or not.
        """
        super().__init__(scale=SPRITE_SCALE)
        self.textures_dict: dict[TextureType, arcade.Texture] = {
            texture: texture.value for texture in texture_types
        }
        self.blocking: bool = blocking

    def __repr__(self: Graphics) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return (
            f"<Graphics (Texture count={len(self.textures_dict)})"
            f" (Blocking={self.blocking})>"
        )


class InstantEffect(GameObjectComponent):
    """Allows a game object to provide an instant effect."""

    __slots__ = ("increase", "level_limit")

    # Class variables
    component_type: ComponentType = ComponentType.INSTANT_EFFECT

    def __init__(
        self: InstantEffect,
        increase: Callable[[int], float],
        level_limit: int,
        **_: D,
    ) -> None:
        """Initialise the object.

        Args:
            increase: The exponential lambda function which calculates the next level's
                value based on the current level.
            level_limit: The max level that this instant effect can be.
        """
        self.increase: Callable[[int], float] = increase
        self.level_limit: int = level_limit

    def __repr__(self: InstantEffect) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<InstantEffect (Level limit={self.level_limit})>"


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

    def __init__(self: Inventory[T], width: int, height: int, **_: D) -> None:
        """Initialise the object.

        Args:
            width: The width of the inventory.
            height: The height of the inventory.
        """
        self.width: int = width
        self.height: int = height
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
            SpaceError: The inventory is empty.
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


class StatusEffect(GameObjectComponent):
    """Allows a game object to provide a status effect."""

    __slots__ = ("increase", "duration", "level_limit")

    # Class variables
    component_type: ComponentType = ComponentType.STATUS_EFFECT

    def __init__(
        self: StatusEffect,
        increase: Callable[[int], float],
        duration: Callable[[int], float],
        level_limit: int,
        **_: D,
    ) -> None:
        """Initialise the object.

        Args:
            increase: The exponential lambda function which calculates the next level's
                value based on the current level.
            duration: The exponential lambda function which calculates the next level's
                duration based on the current level.
            level_limit: The max level that this status effect can be.
        """
        self.increase: Callable[[int], float] = increase
        self.duration: Callable[[int], float] = duration
        self.level_limit: int = level_limit

    def __repr__(self: StatusEffect) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<StatusEffect (Level limit={self.level_limit})>"
