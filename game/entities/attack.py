from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING, Any

# Pip
import arcade

if TYPE_CHECKING:
    from entities.base import Entity
    from physics import PhysicsEngine


# Get the logger
logger = logging.getLogger(__name__)


class Bullet(arcade.SpriteSolidColor):
    """
    Represents a bullet in the game.

    Parameters
    ----------
    x: float
        The starting x position of the bullet.
    y: float
        The starting y position of the bullet.
    width: int
        Width of the bullet.
    height: int
        Height of the bullet.
    color: tuple[int, int, int]
        The color of the bullet.
    owner: Entity
        The entity which shot the bullet.
    """

    def __init__(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        color: tuple[int, int, int],
        owner: Entity,
    ) -> None:
        super().__init__(width, height, color)
        self.center_x: float = x
        self.center_y: float = y
        self.owner: Entity = owner

    def __repr__(self) -> str:
        return f"<Bullet (Position=({self.center_x}, {self.center_y}))>"


class AttackBase:
    """
    The base class for all attack algorithms.

    Parameters
    ----------
    owner: Entity
        The owner of this AI algorithm.
    """

    def __init__(self, owner: Entity) -> None:
        self.owner: Entity = owner

    def __repr__(self) -> str:
        return f"<AttackBase (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """
        Performs an attack by the owner entity.

        Parameters
        ----------
        args: Any
            A tuple containing the parameters needed for the attack.

        Raises
        ------
        NotImplementedError
            The function is not implemented.
        """
        raise NotImplementedError


class RangedAttack(AttackBase):
    """An algorithm which creates a bullet with a set velocity in the direction the
    entity is facing."""

    def __repr__(self) -> str:
        return f"<RangedAttack (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """
        Performs a ranged attack in the direction the entity is facing.

        Parameters
        ----------
        args: Any
            A tuple containing the parameters needed for the attack.
        """
        # Make sure we have the bullet constants. This avoids a circular import
        from constants.entity import BULLET_OFFSET, BULLET_VELOCITY

        # Make sure the needed parameters are valid
        bullet_list: arcade.SpriteList = args[0]

        # Reset the time counter
        self.owner.time_since_last_attack = 0

        # Create and add the new bullet to the physics engine
        new_bullet = Bullet(
            self.owner.center_x,
            self.owner.center_y,
            25,
            5,
            arcade.color.RED,
            self.owner,
        )
        new_bullet.angle = self.owner.direction
        physics: PhysicsEngine = self.owner.physics_engines[0]
        physics.add_bullet(new_bullet)
        bullet_list.append(new_bullet)

        # Move the bullet away from the entity a bit to stop its colliding with them
        angle_radians = self.owner.direction * math.pi / 180
        new_x, new_y = (
            new_bullet.center_x + math.cos(angle_radians) * BULLET_OFFSET,
            new_bullet.center_y + math.sin(angle_radians) * BULLET_OFFSET,
        )
        physics.set_position(new_bullet, (new_x, new_y))

        # Calculate its velocity
        change_x, change_y = (
            math.cos(angle_radians) * BULLET_VELOCITY,
            math.sin(angle_radians) * BULLET_VELOCITY,
        )
        physics.set_velocity(new_bullet, (change_x, change_y))
        logger.info(
            f"Created bullet with owner {self.owner} at position"
            f" ({new_bullet.center_x}, {new_bullet.center_y}) with velocity"
            f" ({change_x}, {change_y})"
        )


class MeleeAttack(AttackBase):
    """"""

    def __repr__(self) -> str:
        return f"<MeleeAttack (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """"""
        raise NotImplementedError


class AreaOfEffectAttack(AttackBase):
    """"""

    def __repr__(self) -> str:
        return f"<AreaOfEffectAttack (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """"""
        raise NotImplementedError
