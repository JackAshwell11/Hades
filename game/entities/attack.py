from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING, Any

# Pip
import arcade

if TYPE_CHECKING:
    from game.constants.entity import AreaOfEffectAttackData, AttackData
    from game.entities.base import Entity
    from game.physics import PhysicsEngine


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
    damage: int
        The damage this bullet deals.
    """

    def __init__(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        color: tuple[int, int, int],
        owner: Entity,
        damage: int,
    ) -> None:
        super().__init__(width, height, color)
        self.center_x: float = x
        self.center_y: float = y
        self.owner: Entity = owner
        self.damage: int = damage

    def __repr__(self) -> str:
        return f"<Bullet (Position=({self.center_x}, {self.center_y}))>"


class AttackBase:
    """
    The base class for all attack algorithms.

    Parameters
    ----------
    owner: Entity
        The owner of this attack algorithm.
    attack_data: AttackData
        The entity data about this attack.
    """

    def __init__(
        self,
        owner: Entity,
        attack_data: AttackData,
    ) -> None:
        self.owner: Entity = owner
        self.attack_data: AttackData = attack_data

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
    """
    An algorithm which creates a bullet with a set velocity in the direction the
    entity is facing.

    Parameters
    ----------
    owner: Entity
        The owner of this attack algorithm.
    attack_data: AttackData
        The entity data about this attack.
    """

    def __init__(self, owner: Entity, attack_data: AttackData) -> None:
        super().__init__(owner, attack_data)

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
        from game.constants.entity import BULLET_OFFSET, BULLET_VELOCITY

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
            self.attack_data.damage,
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
    """
    DO

    Parameters
    ----------
    owner: Entity
        The owner of this attack algorithm.
    attack_data: AttackData
        The entity data about this attack.
    """

    def __init__(self, owner: Entity, attack_data: AttackData) -> None:
        super().__init__(owner, attack_data)

    def __repr__(self) -> str:
        return f"<MeleeAttack (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """"""
        raise NotImplementedError


class AreaOfEffectAttack(AttackBase):
    """
    An algorithm which creates an area around the entity with a set radius and deals
    damage to any entities that are within that range.

    Parameters
    ----------
    owner: Entity
        The owner of this attack algorithm.
    attack_data: AreaOfEffectAttackData
        The entity data about this attack.
    """

    def __init__(self, owner: Entity, attack_data: AreaOfEffectAttackData) -> None:
        super().__init__(owner, attack_data)

    def __repr__(self) -> str:
        return f"<AreaOfEffectAttack (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """"""
        # Make sure we have the sprite size. This avoids a circular import
        from game.constants.entity import SPRITE_SIZE

        # Make sure the needed parameters are valid
        target_entity: arcade.SpriteList | arcade.Sprite = args[0]

        # Create a sprite with an empty texture
        empty_texture = arcade.Texture.create_empty(
            "",
            (
                int(self.attack_data.area_of_effect_range * 2 * SPRITE_SIZE),
                int(self.attack_data.area_of_effect_range * 2 * SPRITE_SIZE),
            ),
        )
        area_of_effect_sprite = arcade.Sprite(
            center_x=self.owner.center_x,
            center_y=self.owner.center_y,
            texture=empty_texture,
        )

        # Detect a collision/collisions between the area_of_effect_sprite and the target
        try:
            if arcade.check_for_collision(area_of_effect_sprite, target_entity):
                # Target is the player so deal damage
                target_entity.deal_damage(self.attack_data.damage)  # noqa
            return
        except TypeError:
            for entity in arcade.check_for_collision_with_list(
                area_of_effect_sprite, target_entity
            ):
                # Deal damage to all the enemies within range
                entity.deal_damage(self.attack_data.damage)  # noqa
