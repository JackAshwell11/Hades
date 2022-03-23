from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from constants import ENEMY_DAMAGE, PLAYER_DAMAGE
from entities.base import EntityID

if TYPE_CHECKING:
    from entities.base import Bullet, Entity
    from entities.player import Player
    from entities.tile import Tile


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
    except AttributeError:
        # An error randomly occurs here so just ignore it
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
            enemy.deal_damage(PLAYER_DAMAGE)
    except AttributeError:
        # An error randomly occurs here so just ignore it
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
            player.deal_damage(ENEMY_DAMAGE)
    except AttributeError:
        # An error randomly occurs here so just ignore it
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
        wall_list: arcade.SpriteList,
        enemy_list: arcade.SpriteList,
    ) -> None:
        """
        Setups up the various sprites needed for the physics engine to work properly.

        Parameters
        ----------
        player: Entity
            The player entity.
        wall_list: arcade.SpriteList
            The sprite list for the wall sprites.
        enemy_list: arcade.SpriteList
            The sprite list for the enemy sprites
        """
        # Add the player sprite to the physics engine
        self.add_sprite(
            player,
            moment_of_inertia=arcade.PymunkPhysicsEngine.MOMENT_INF,
            damping=self.damping,
            collision_type="player",
        )

        # Add the wall sprites to the physics engine
        self.add_sprite_list(
            wall_list,
            body_type=arcade.PymunkPhysicsEngine.STATIC,
            collision_type="wall",
        )

        # Add the enemy sprites to the physics engine
        self.add_sprite_list(
            enemy_list,
            moment_of_intertia=arcade.PymunkPhysicsEngine.MOMENT_INF,
            damping=self.damping,
            collision_type="enemy",
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
