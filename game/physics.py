from __future__ import annotations

# Builtin
from typing import TYPE_CHECKING

# Pip
import arcade

# Custom
from entities.character import Bullet

if TYPE_CHECKING:
    from entities.entity import Entity


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
