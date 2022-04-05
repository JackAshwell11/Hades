from __future__ import annotations

# Builtin
import logging
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from constants.enums import EntityID

if TYPE_CHECKING:
    from entities.base import Bullet, Entity
    from entities.player import Player
    from entities.tile import Tile

# Get the logger
logger = logging.getLogger(__name__)


def wall_bullet_begin_handler(wall: Tile, bullet: Bullet, *_) -> bool:
    """
    Handles collision between a wall tile and a bullet sprite as they touch. This uses
    the begin_handler which processes collision when two shapes are touching for the
    first time.

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
        logger.debug(f"Removed {bullet} after hitting {wall}")
    except AttributeError:
        # An error randomly occurs here so just ignore it
        logger.warning(
            f"An error occurred while removing {bullet} after hitting {wall}"
        )
        pass
    # Stop collision processing
    return False


def enemy_bullet_begin_handler(enemy: Entity, bullet: Bullet, *_) -> bool:
    """
    Handles collision between an enemy entity and a bullet sprite as they touch. This
    uses the begin_handler which processes collision when two shapes are touching for
    the first time.

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
        # Remove the bullet
        bullet.remove_from_sprite_lists()

        # Check if the owner is the player
        if bullet.owner.ID is EntityID.PLAYER:
            # Deal damage to the enemy
            enemy.deal_damage(bullet.owner.entity_type.damage)
            logger.debug(f"Removed {bullet} after hitting {enemy}")
    except AttributeError:
        # An error randomly occurs here so just ignore it
        logger.warning(
            f"An error occurred while removing {bullet} after hitting {enemy}"
        )
        pass
    # Stop collision processing
    return False


def player_bullet_begin_handler(player: Player, bullet: Bullet, *_) -> bool:
    """
    Handles collision between a player entity and a bullet sprite as they touch. This
    uses the begin_handler which processes collision when two shapes are touching for
    the first time.

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
        # Remove the bullet
        bullet.remove_from_sprite_lists()

        # Check if the owner is an enemy
        if bullet.owner.ID is EntityID.ENEMY:
            # Deal damage to the player
            player.deal_damage(bullet.owner.entity_type.damage)
            logger.debug(f"Removed {bullet} after hitting {player}")
    except AttributeError:
        # An error randomly occurs here so just ignore it
        logger.warning(
            f"An error occurred while removing {bullet} after hitting {player}"
        )
        pass
    # Stop collision processing
    return False


class PhysicsEngine(arcade.PymunkPhysicsEngine):
    """
    An abstracted version of the Pymunk Physics Engine which eases setting up a physics
    engine for a top-down game.

    Parameters
    ----------
    damping: float
        The amount of speed which is kept to the next tick. A value of 1.0 means no
        speed is lost, while 0.9 means 10% of speed is lost.
    """

    def __init__(self, damping: float) -> None:
        super().__init__(damping=damping)
        self.damping: float = damping

    def setup(
        self,
        player: Entity,
        tile_list: arcade.SpriteList,
        enemy_list: arcade.SpriteList,
    ) -> None:
        """
        Setups up the various sprites needed for the physics engine to work properly.

        Parameters
        ----------
        player: Entity
            The player entity.
        tile_list: arcade.SpriteList
            The sprite list for the tile sprites. This includes both static and
            non-static sprites.
        enemy_list: arcade.SpriteList
            The sprite list for the enemy sprites
        """
        # Add the player sprite to the physics engine
        self.add_sprite(
            player,
            moment_of_inertia=self.MOMENT_INF,
            collision_type="player",
            max_velocity=player.entity_type.max_velocity,
        )

        # Add the static tile sprites to the physics engine
        for tile in tile_list:
            if tile.is_static:  # noqa
                self.add_sprite(
                    tile,
                    body_type=self.STATIC,
                    collision_type="wall",
                )

        # Add the enemy sprites to the physics engine
        for enemy in enemy_list:
            self.add_sprite(
                enemy,
                moment_of_inertia=self.MOMENT_INF,
                collision_type="enemy",
                max_velocity=enemy.entity_type.max_velocity,  # noqa
            )

        # Add collision handlers
        self.add_collision_handler(
            "wall", "bullet", begin_handler=wall_bullet_begin_handler
        )
        self.add_collision_handler(
            "enemy", "bullet", begin_handler=enemy_bullet_begin_handler
        )
        self.add_collision_handler(
            "player", "bullet", begin_handler=player_bullet_begin_handler
        )
        logger.info(f"Initialised physics engine with {len(self.sprites.keys())} items")

    def __repr__(self) -> str:
        return (
            f"<PhysicsEngine (Damping={self.damping}) (Sprite"
            f" count={len(self.sprites)})>"
        )

    def add_bullet(self, bullet: Bullet) -> None:
        """
        Adds a bullet to the physics engine.

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
        logger.info(f"Added bullet {bullet} to physics engine")
