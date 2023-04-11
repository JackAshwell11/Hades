"""Manages the physics using an abstracted version of the Pymunk physics engine."""
from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from hades.constants.game_objects import ObjectID

if TYPE_CHECKING:
    from hades.game_objects_OLD.attacks import Bullet
    from hades.game_objects_OLD.base import Entity, Tile
    from hades.game_objects_OLD.enemies import Enemy
    from hades.game_objects_OLD.players import Player
    from pymunk.arbiter import Arbiter
    from pymunk.space import Space

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

    Parameters
    ----------
    wall: Tile
        The wall tile which the bullet hit.
    bullet: Bullet
        The bullet sprite which hit the wall tile.

    Returns
    -------
    bool
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

    Parameters
    ----------
    enemy: Entity
        The enemy entity which the bullet hit.
    bullet: Bullet
        The bullet sprite which hit the enemy entity.

    Returns
    -------
    bool
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

    Parameters
    ----------
    player: Player
        The player entity which the bullet hit.
    bullet: Bullet
        The bullet sprite which hit the enemy entity.

    Returns
    -------
    bool
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
    """A class which eases setting up the Pymunk physics engine for a top-down game."""

    def __init__(self, damping: float) -> None:
        """Initialise the object.

        Parameters
        ----------
        damping: float
            The amount of speed which is kept to the next tick. A value of 1.0 means no
            speed is lost, while 0.9 means 10% of speed is lost.
        """
        super().__init__(damping=damping)
        self.damping: float = damping

    def setup(
        self,
        player: Entity,
        tile_list: arcade.SpriteList,
    ) -> None:
        """Set-ups the various sprites needed for the physics engine to work properly.

        Parameters
        ----------
        player: Entity
            The player entity.
        tile_list: arcade.SpriteList
            The sprite list for the tile sprites. This includes both static and
            non-static sprites.
        """
        # Add the player sprite to the physics engine
        self.add_sprite(
            player,
            moment_of_inertia=self.MOMENT_INF,
            collision_type="player",
            max_horizontal_velocity=int(player.max_velocity.value),
            max_vertical_velocity=int(player.max_velocity.value),
        )
        logger.debug("Added %r to physics engine", player)

        # Add the static tile sprites to the physics engine
        for tile in tile_list:  # type: Tile
            if tile.blocking:
                self.add_sprite(
                    tile,
                    body_type=self.STATIC,
                    collision_type="wall",
                )
            logger.debug("Added %r to physics engine", tile)

        # Add collision handlers
        self.add_collision_handler(
            "wall",
            "bullet",
            begin_handler=wall_bullet_begin_handler,
        )
        self.add_collision_handler(
            "enemy",
            "bullet",
            begin_handler=enemy_bullet_begin_handler,
        )
        self.add_collision_handler(
            "player",
            "bullet",
            begin_handler=player_bullet_begin_handler,
        )
        logger.info(
            "Initialised physics engine with %d items",
            len(self.sprites.keys()),
        )

    def add_bullet(self, bullet: Bullet) -> None:
        """Add a bullet to the physics engine.

        Parameters
        ----------
        bullet: Bullet
            The bullet to add to the physics engine.
        """
        self.add_sprite(
            bullet,
            moment_of_inertia=self.MOMENT_INF,
            body_type=self.KINEMATIC,
            collision_type="bullet",
        )
        logger.debug("Added bullet %r to physics engine", bullet)

    def add_enemy(self, enemy: Enemy) -> None:
        """Add an enemy to the physics engine.

        Parameters
        ----------
        enemy: Enemy
            The enemy to add to the physics engine.
        """
        self.add_sprite(
            enemy,
            moment_of_inertia=self.MOMENT_INF,
            collision_type="enemy",
            max_horizontal_velocity=int(enemy.max_velocity.value),
            max_vertical_velocity=int(enemy.max_velocity.value),
        )
        logger.debug("Added %r to physics engine", enemy)

    def __repr__(self) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return (
            f"<PhysicsEngine (Damping={self.damping}) (Sprite"
            f" count={len(self.sprites)})>"
        )
