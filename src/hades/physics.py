"""Manages the physics using an abstracted version of the Pymunk physics engine."""

from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
from arcade import PymunkPhysicsEngine

# Custom
from hades.constants import DAMPING, MAX_VELOCITY

if TYPE_CHECKING:
    from hades.constants import GameObjectType
    from hades.sprite import HadesSprite

__all__ = ("PhysicsEngine",)

# Get the logger
logger = logging.getLogger(__name__)


class PhysicsEngine(PymunkPhysicsEngine):
    """Eases setting up the Pymunk physics engine for a top-down game."""

    def __init__(self: PhysicsEngine) -> None:
        """Initialise the object."""
        super().__init__(damping=DAMPING)

    def add_game_object(
        self: PhysicsEngine,
        sprite: HadesSprite,
        game_object_type: GameObjectType,
        *,
        blocking: bool,
    ) -> None:
        """Add a game object to the physics engine.

        Args:
            sprite: The sprite to add to the physics engine.
            game_object_type: The type of the game object.
            blocking: Whether the game object blocks sprite movement or not.
        """
        logger.debug(
            "Adding %r game object %r to the physics engine",
            blocking,
            sprite,
        )
        self.add_sprite(
            sprite,
            moment_of_inertia=None if blocking else self.MOMENT_INF,
            body_type=self.STATIC if blocking else self.DYNAMIC,
            max_velocity=MAX_VELOCITY,
            collision_type=game_object_type.name,
        )

    def __repr__(self: PhysicsEngine) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<PhysicsEngine (Sprite count={len(self.sprites)})>"
