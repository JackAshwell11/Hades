"""Manages the entity component system and its processes."""
from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hades.game_objects.base import ComponentType, Processor

__all__ = ("EntityComponentSystem",)


class EntityComponentSystem:
    """Stores and manages the different game objects registered with the system."""

    def __init__(self: EntityComponentSystem) -> None:
        """Initialise the object."""
        self._game_object_signatures: dict[int, ComponentType] = {}
        self._processors: dict[ComponentType, Processor] = {}
        self._components: dict[ComponentType, set[int]] = {}

        # TODO: IF GO FIRST METHOD, THEN GMAE_OBJECT_SIGNATURES NEEDS CHANGING TO
        #  dict[int, set[GameObjectComponent]] WITH NAME CHANGE AND DELETE PROCESSORS

    def __repr__(self: EntityComponentSystem) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return (
            "<EntityComponentSystem (Game object"
            f" count={len(self._game_object_signatures)}) (Processor"
            f" count={len(self._processors)})>"
        )


system = EntityComponentSystem()
print(system)
