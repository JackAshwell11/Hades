from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING, Any

# Pip
import arcade

if TYPE_CHECKING:
    from game.constants.entity import (
        AreaOfEffectAttackData,
        MeleeAttackData,
        RangedAttackData,
    )
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
    max_range: float
        The max range of the bullet.

    Attributes
    ----------
    start_position: tuple[float, float]
        The starting position of the bullet. This is used to kill the bullet after a
        certain amount of tiles if it hasn't hit anything.
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
        max_range: float,
    ) -> None:
        super().__init__(width, height, color)
        self.center_x: float = x
        self.center_y: float = y
        self.owner: Entity = owner
        self.damage: int = damage
        self.max_range: float = max_range
        self.start_position: tuple[float, float] = self.center_x, self.center_y

    def __repr__(self) -> str:
        return f"<Bullet (Position=({self.center_x}, {self.center_y}))>"

    def on_update(self, delta_time: float = 1 / 60) -> None:
        """
        Processes bullet logic.

        Parameters
        ----------
        delta_time: float
            Time interval since the last time the function was called.
        """
        # Check if the bullet passed the max range
        if (
            math.hypot(
                self.center_x - self.start_position[0],
                self.center_y - self.start_position[1],
            )
            >= self.max_range
        ):
            self.remove_from_sprite_lists()
            logger.debug(f"Removed {self} after passing max_range {self.max_range}")


class AttackBase:
    """
    The base class for all attack algorithms.

    Parameters
    ----------
    owner: Entity
        The owner of this attack algorithm.
    attack_cooldown: int
        The cooldown for this attack.
    """

    __slots__ = (
        "owner",
        "attack_cooldown",
    )

    def __init__(self, owner: Entity, attack_cooldown: int) -> None:
        self.owner: Entity = owner
        self.attack_cooldown: int = attack_cooldown

    def __repr__(self) -> str:
        return f"<AttackBase (Owner={self.owner})>"

    @property
    def attack_range(self) -> int:
        """
        Gets the attack range for this attack.

        Returns
        -------
        int
            The attack range for this attack.
        """
        raise NotImplementedError

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
    attack_cooldown: int
        The cooldown for this attack.
    """

    __slots__ = ()

    def __init__(self, owner: Entity, attack_cooldown: int) -> None:
        super().__init__(owner, attack_cooldown)

    def __repr__(self) -> str:
        return f"<RangedAttack (Owner={self.owner})>"

    @property
    def ranged_attack_data(self) -> RangedAttackData:
        """
        Gets the ranged attack data.

        Returns
        -------
        RangedAttackData
            The ranged attack data.
        """
        return self.owner.ranged_attack_data

    @property
    def attack_range(self) -> int:
        """
        Gets the attack range for this attack.

        Returns
        -------
        int
            The attack range for this attack.
        """
        return self.ranged_attack_data.attack_range

    def process_attack(self, *args: Any) -> None:
        """
        Performs a ranged attack in the direction the entity is facing.

        Parameters
        ----------
        args: Any
            A tuple containing the parameters needed for the attack.
        """
        # Make sure we have the bullet constants. This avoids a circular import
        from game.constants.entity import BULLET_VELOCITY, SPRITE_SIZE

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
            self.ranged_attack_data.damage,
            self.ranged_attack_data.max_range * SPRITE_SIZE,
        )
        new_bullet.angle = self.owner.direction
        physics: PhysicsEngine = self.owner.physics_engines[0]
        physics.add_bullet(new_bullet)
        bullet_list.append(new_bullet)

        # Calculate its velocity
        angle_radians = self.owner.direction * math.pi / 180
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
    An algorithm which performs a melee attack in the direction the entity is looking
    dealing damage to any entity that is within a specific angle range of the entity's
    direction and are within the attack distance. Since when the enemy is attacking,
    they are always facing the player, we don't need to do the angle range check for
    enemies.

    Parameters
    ----------
    owner: Entity
        The owner of this attack algorithm.
    attack_cooldown: int
        The cooldown for this attack.
    """

    __slots__ = ()

    def __init__(self, owner: Entity, attack_cooldown: int) -> None:
        super().__init__(owner, attack_cooldown)

    def __repr__(self) -> str:
        return f"<MeleeAttack (Owner={self.owner})>"

    @property
    def melee_attack_data(self) -> MeleeAttackData:
        """
        Gets the melee attack data.

        Returns
        -------
        MeleeAttackData
            The melee attack data.
        """
        return self.owner.melee_attack_data

    @property
    def attack_range(self) -> int:
        """
        Gets the attack range for this attack.

        Returns
        -------
        int
            The attack range for this attack.
        """
        return self.melee_attack_data.attack_range

    def process_attack(self, *args: Any) -> None:
        """
        Performs a melee attack in the direction the entity is facing.

        Parameters
        ----------
        args: Any
            A tuple containing the parameters needed for the attack.
        """
        # Make sure the needed parameters are valid
        targets: list[Entity] = args[0]

        # Deal damage to all entities within range
        for entity in targets:
            entity.deal_damage(self.melee_attack_data.damage)


class AreaOfEffectAttack(AttackBase):
    """
    An algorithm which creates an area around the entity with a set radius and deals
    damage to any entities that are within that range.

    Parameters
    ----------
    owner: Entity
        The owner of this attack algorithm.
    attack_cooldown: int
        The cooldown for this attack.
    """

    __slots__ = ()

    def __init__(self, owner: Entity, attack_cooldown: int) -> None:
        super().__init__(owner, attack_cooldown)

    def __repr__(self) -> str:
        return f"<AreaOfEffectAttack (Owner={self.owner})>"

    @property
    def area_of_effect_attack_data(self) -> AreaOfEffectAttackData:
        """
        Gets the area of effect attack data.

        Returns
        -------
        AreaOfEffectAttackData
            The area of effect attack data.
        """
        return self.owner.area_of_effect_attack_data

    @property
    def attack_range(self) -> int:
        """
        Gets the attack range for this attack.

        Returns
        -------
        int
            The attack range for this attack.
        """
        return self.area_of_effect_attack_data.attack_range

    def process_attack(self, *args: Any) -> None:
        """
        Performs an area of effect attack around the entity.

        Parameters
        ----------
        args: Any
            A tuple containing the parameters needed for the attack.
        """
        # Make sure we have the sprite size. This avoids a circular import
        from game.constants.entity import SPRITE_SIZE

        # Make sure the needed parameters are valid
        target_entity: arcade.SpriteList | Entity = args[0]

        # Create a sprite with an empty texture
        empty_texture = arcade.Texture.create_empty(
            "",
            (
                int(self.attack_range * 2 * SPRITE_SIZE),
                int(self.attack_range * 2 * SPRITE_SIZE),
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
                target_entity.deal_damage(self.area_of_effect_attack_data.damage)
            return
        except TypeError:
            for entity in arcade.check_for_collision_with_list(
                area_of_effect_sprite, target_entity
            ):
                # Deal damage to all the enemies within range
                entity.deal_damage(self.area_of_effect_attack_data.damage)  # noqa
