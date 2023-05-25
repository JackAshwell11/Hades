"""Manages the physics using an abstracted version of the Pymunk physics engine."""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from hades.constants import DAMPING, MAX_VELOCITY
from hades.constants_OLD.game_objects import ObjectID

if TYPE_CHECKING:
    from pymunk.arbiter import Arbiter
    from pymunk.space import Space

    from hades.game_objects.sprite import HadesSprite

__all__ = ("PhysicsEngine",)

# Get the logger
logger = logging.getLogger(__name__)


def wall_bullet_begin_handler(
    wall: Tile,
    bullet: Bullet,
    *_: tuple[Arbiter, Space],
) -> bool:
    """Handle collision between a wall tile and a bullet sprite as they touch.

    This uses the begin_handler which processes collision when two shapes are touching
    for the first time.

    Args:
        wall: The wall tile which the bullet hit.
        bullet: The bullet sprite which hit the wall tile.

    Returns:
        Whether Pymunk should process the collision or not. This handler returns False
        since we just want to remove the bullet and not process collision.
    """
    try:
        # Remove the bullet
        bullet.remove_from_sprite_lists()
        logger.debug("Removed %r after hitting %r", bullet, wall)
    except AttributeError:
        # An error randomly occurs here so just ignore it
        logger.warning(
            "An error occurred while removing %r after hitting %r",
            bullet,
            wall,
        )
    # Stop collision processing
    return False


def enemy_bullet_begin_handler(
    enemy: Entity,
    bullet: Bullet,
    *_: tuple[Arbiter, Space],
) -> bool:
    """Handle collision between an enemy entity and a bullet sprite as they touch.

    This uses the begin_handler which processes collision when two shapes are touching
    for the first time.

    Args:
        enemy: The enemy entity which the bullet hit.
        bullet: The bullet sprite which hit the enemy entity.

    Returns:
        Whether Pymunk should process the collision or not. This handler returns False
        since we just want to remove the bullet and not process collision.
    """
    try:
        # Check if the owner is the player
        if bullet.owner.object_id is ObjectID.PLAYER:
            # Remove the bullet
            bullet.remove_from_sprite_lists()

            # Deal damage to the enemy
            enemy.deal_damage(bullet.damage)
            logger.debug("Removed %r after hitting %r", bullet, enemy)
    except AttributeError:
        # An error randomly occurs here so just ignore it
        logger.warning(
            "An error occurred while removing %r after hitting %r",
            bullet,
            enemy,
        )
    # Stop collision processing
    return False


def player_bullet_begin_handler(
    player: Player,
    bullet: Bullet,
    *_: tuple[Arbiter, Space],
) -> bool:
    """Handle collision between a player entity and a bullet sprite as they touch.

    This uses the begin_handler which processes collision when two shapes are touching
    for the first time.

    Args:
        player: The player entity which the bullet hit.
        bullet: The bullet sprite which hit the enemy entity.

    Returns:
        Whether Pymunk should process the collision or not. This handler returns False
        since we just want to remove the bullet and not process collision.
    """
    try:
        # Check if the owner is an enemy
        if bullet.owner.object_id is ObjectID.ENEMY:
            # Remove the bullet
            bullet.remove_from_sprite_lists()

            # Deal damage to the player
            player.deal_damage(bullet.damage)
            logger.debug("Removed %r after hitting %r", bullet, player)
    except AttributeError:
        # An error randomly occurs here so just ignore it
        logger.warning(
            "An error occurred while removing %r after hitting %r",
            bullet,
            player,
        )
    # Stop collision processing
    return False


class PhysicsEngine(arcade.PymunkPhysicsEngine):
    """Eases setting up the Pymunk physics engine for a top-down game."""

    def __init__(self: PhysicsEngine) -> None:
        """Initialise the object."""
        super().__init__(damping=DAMPING)

        # TODO: Add collision handlers

    def add_game_object(
        self: PhysicsEngine,
        sprite: HadesSprite,
        *,
        blocking: bool,
    ) -> None:
        """Add a game object to the physics engine.

        Args:
            sprite: The sprite to add to the physics engine.
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
            max_horizontal_velocity=MAX_VELOCITY,
            max_vertical_velocity=MAX_VELOCITY,
            collision_type=sprite.game_object_type.name,
        )

    def __repr__(self: PhysicsEngine) -> str:
        """Return a human-readable representation of this object.

        Returns:
            The human-readable representation of this object.
        """
        return f"<PhysicsEngine (Sprite count={len(self.sprites)})>"
