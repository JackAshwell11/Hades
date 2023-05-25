"""Stores the foundations for the entity component system and its functionality."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

    from hades.constants import ComponentType
    from hades.game_objects.system import ECS

__all__ = (
    "ComponentData",
    "GameObjectComponent",
)


class ComponentData(TypedDict, total=False):
    """Holds the data needed to initialise the components."""

    armour_regen: bool
    attributes: dict[ComponentType, tuple[int, int]]
    instant_effects: tuple[int, dict[ComponentType, Callable[[int], float]]]
    inventory_size: tuple[int, int]
    status_effects: tuple[
        int,
        dict[ComponentType, tuple[Callable[[int], float], Callable[[int], float]]],
    ]


class GameObjectComponent:
    """The base class for all game object components."""

    __slots__ = ("game_object_id", "system")

    # Class variables
    component_type: ComponentType

    def __init__(
        self: GameObjectComponent,
        game_object_id: int,
        system: ECS,
        _: ComponentData,
    ) -> None:
        """Initialise the object.

        Args:
            game_object_id: The game object ID.
            system: The entity component system which manages the game objects.
        """
        self.game_object_id: int = game_object_id
        self.system: ECS = system


# Record which component types are entity attributes
