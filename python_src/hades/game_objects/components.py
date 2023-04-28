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

    from arcade import SpriteList

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

        Parameters
        ----------
        full: bool
            Whether the inventory is empty or full.
        """
        super().__init__(f"The inventory is {'full' if full else 'empty'}.")


class Graphics(arcade.Sprite, GameObjectComponent):
    """Allows a game object to be drawn on the screen and interact with Arcade.

    Attributes
    ----------
    textures: dict[TextureType, arcade.Texture]
        The textures which represent this game object.
    """

    # Class variables
    component_type: ComponentType = ComponentType.GRAPHICS

    def __init__(
        self: Graphics,
        texture_types: set[TextureType],
        *,
        blocking: bool = False,
    ) -> None:
        """Initialise the object.

        Parameters
        ----------
        texture_types: set[TextureType]
            The
        blocking: bool
            Whether this component is blocking or not.
        """
        super().__init__(scale=SPRITE_SCALE)
        self.textures_dict: dict[TextureType, arcade.Texture] = {
            texture: texture.value for texture in texture_types
        }
        self.blocking: bool = blocking

    def add_to_spritelist(self: Graphics, spritelist: SpriteList) -> Graphics:
        """Add the graphics component to a given spritelist.

        Parameters
        ----------
        spritelist: SpriteList
            The arcade spritelist to add the graphics component to.

        Returns
        -------
        Graphics
            The graphics component.
        """
        spritelist.append(self)
        return self

    def __repr__(self: Graphics) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
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
    ) -> None:
        """Initialise the object.

        Parameters
        ----------
        increase: Callable[[int], float]
            The exponential lambda function which calculates the next level's value
            based on the current level.
        level_limit: int
            The max level that this instant effect can be.
        """
        self.increase: Callable[[int], float] = increase
        self.level_limit: int = level_limit

    def __repr__(self: InstantEffect) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<InstantEffect (Level limit={self.level_limit})>"


class Inventory(Generic[T], GameObjectComponent):
    """Allows a game object to have a fixed size inventory.

    Attributes
    ----------
    inventory: list[T]
        The game object's inventory.
    """

    __slots__ = (
        "width",
        "height",
        "inventory",
    )

    # Class variables
    component_type: ComponentType = ComponentType.INVENTORY

    def __init__(self: Inventory[T], width: int, height: int) -> None:
        """Initialise the object.

        Parameters
        ----------
        width: int
            The width of the inventory.
        height: int
            The height of the inventory.
        """
        self.width: int = width
        self.height: int = height
        self.inventory: list[T] = []

    def add_item_to_inventory(self: Inventory[T], item: T) -> None:
        """Add an item to the inventory.

        Parameters
        ----------
        item: T
            The item to add to the inventory.

        Raises
        ------
        SpaceError
            The inventory container does not have enough room.
        """
        if len(self.inventory) == self.width * self.height:
            raise InventorySpaceError(full=True)
        self.inventory.append(item)

    def remove_item_from_inventory(self: Inventory[T], index: int) -> T:
        """Remove an item at a specific index.

        Parameters
        ----------
        index: int
            The index to remove an item at.

        Raises
        ------
        SpaceError
            The inventory container does not have enough room.

        Returns
        -------
        T
            The item at position `index` in the inventory.
        """
        if len(self.inventory) < index:
            raise InventorySpaceError(full=False)
        return self.inventory.pop(index)

    def __repr__(self: Inventory[T]) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
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
    ) -> None:
        """Initialise the object.

        Parameters
        ----------
        increase: Callable[[int], float]
            The exponential lambda function which calculates the next level's value
            based on the current level.
        duration: Callable[[int], float]
            The exponential lambda function which calculates the next level's duration
            based on the current level.
        level_limit: int
            The max level that this status effect can be.
        """
        self.increase: Callable[[int], float] = increase
        self.duration: Callable[[int], float] = duration
        self.level_limit: int = level_limit

    def __repr__(self: StatusEffect) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<StatusEffect (Level limit={self.level_limit})>"
