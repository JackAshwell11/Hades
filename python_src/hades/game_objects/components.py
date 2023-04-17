"""Manages various components available to the game objects."""
from __future__ import annotations

# Custom
from hades.exceptions import SpaceError
from hades.game_objects.base import ComponentType, GameObjectComponent

__all__ = ("Inventory",)


class Inventory(GameObjectComponent):
    """Allows a game object to have a fixed size inventory.

    Attributes
    ----------
    inventory: list[int]
        The game object's inventory.
    """

    __slots__ = (
        "width",
        "height",
        "inventory",
    )

    # Class variables
    component_type: ComponentType = ComponentType.INVENTORY

    def __init__(self: Inventory, width: int, height: int) -> None:
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
        self.inventory: list[int] = []

    def add_item_to_inventory(self: Inventory, item: int) -> None:
        """Add an item to the inventory.

        Parameters
        ----------
        item: int
            The item to add to the inventory.

        Raises
        ------
        SpaceError
            The inventory container does not have enough room
        """
        if len(self.inventory) == self.width * self.height:
            raise SpaceError(self.__class__.__name__.lower())
        self.inventory.append(item)

    def remove_item_from_inventory(self: Inventory, index: int) -> int:
        """Remove an item at a specific index.

        Parameters
        ----------
        index: int
            The index to remove an item at.


        Raises
        ------
        SpaceError
            The inventory container does not have enough room

        Returns
        -------
        ValueError
            Not enough space in the inventory
        """
        if len(self.inventory) < index:
            raise SpaceError(self.__class__.__name__.lower())
        return self.inventory.pop(index)

    def __repr__(self: Inventory) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return f"<Inventory (Width={self.width}) (Height={self.height})>"


# TODO: So system will have collection of game objects which are represented with a dict
#  with key being component enum and value being instantiated component. Arcade.sprite
#  should be a graphics component and will be added to a spritelist on initialisation.
#  Game objects can be put into groups inside system (entities, tiles, particles). USE
#  https://github.com/avikor/entity_component_system/tree/master/ecs AND
#  https://github.com/benmoran56/esper/blob/master/esper/__init__.py (MAINLY THIS ONE)

# TODO: DETERMINE HOW TO STORE COMPONENTS AND GAME OBJECTS. SHOULD PROCESSORS BE USED?
#  SHOULD GAMEOBJECTCOMPONENT BE USED? SHOULD _COMPONENTS AND _GAME_OBJECTS BE USED?
