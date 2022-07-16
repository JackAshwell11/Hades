"""Stores the different attack algorithms that are available to the entities."""
from __future__ import annotations

# Builtin
import logging
import math
from typing import TYPE_CHECKING, Any

# Pip
import arcade

# Custom
from game.constants.game_object import BULLET_VELOCITY, SPRITE_SIZE, AttackAlgorithmType

if TYPE_CHECKING:
    from game.constants.game_object import AttackData
    from game.game_objects.base import Entity
    from game.physics import PhysicsEngine

__all__ = (
    "AreaOfEffectAttack",
    "AttackBase",
    "Bullet",
    "MeleeAttack",
    "RangedAttack",
    "create_attack",
)

# Get the logger
logger = logging.getLogger(__name__)


class Bullet(arcade.SpriteSolidColor):
    """Represents a bullet in the game.

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
        self.angle: float = owner.direction

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<Bullet (Position=({self.center_x}, {self.center_y}))>"

    def on_update(self, _: float = 1 / 60) -> None:
        """Process bullet logic."""
        # Check if the bullet is pass the max range
        if math.dist(self.position, self.start_position) >= self.max_range:
            self.remove_from_sprite_lists()
            logger.debug("Removed %r after passing max range %f", self, self.max_range)


class AttackBase:
    """The base class for all attack algorithms.

    Parameters
    ----------
    owner: Entity
        The reference to the enemy object that manages this attack algorithm.
    attack_data: AttackData
        The data for this attack.
    """

    # Class variables
    attack_type: AttackAlgorithmType = AttackAlgorithmType.BASE

    __slots__ = (
        "owner",
        "attack_data",
    )

    def __init__(self, owner: Entity, attack_data: AttackData) -> None:
        self.owner: Entity = owner
        self.attack_data: AttackData = attack_data

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<AttackBase (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """Perform an attack by the owner entity.

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
    """Creates a bullet in the direction the entity is facing with a set velocity."""

    # Class variables
    attack_type: AttackAlgorithmType = AttackAlgorithmType.RANGED

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<RangedAttack (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """Perform a ranged attack in the direction the entity is facing.

        Parameters
        ----------
        args: Any
            A tuple containing the parameters needed for the attack.
        """
        # Make sure variables needed are valid
        assert self.attack_data.extra is not None

        # Make sure the needed parameters are valid
        bullet_list: arcade.SpriteList = args[0]
        logger.info("Entity %r is performing a ranged attack", self.owner)

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
            self.attack_data.extra.max_bullet_range * SPRITE_SIZE,
        )
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
        logger.debug(
            "Created bullet with owner %r at position %r with velocity %r",
            self.owner,
            new_bullet.position,
            (change_x, change_y),
        )


class MeleeAttack(AttackBase):
    """Performs a melee attack dealing damage to any entity in front of the owner."""

    # Class variables
    attack_type: AttackAlgorithmType = AttackAlgorithmType.MELEE

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<MeleeAttack (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """Perform a melee attack in the direction the entity is facing.

        Parameters
        ----------
        args: Any
            A tuple containing the parameters needed for the attack.
        """
        # Make sure the needed parameters are valid
        targets: list[Entity] = args[0]
        logger.info(
            "Entity %r is performing a melee attack on targets %r", self.owner, targets
        )

        # Deal damage to all entities within range
        for entity in targets:
            entity.deal_damage(self.attack_data.damage)


class AreaOfEffectAttack(AttackBase):
    """Creates an area around the entity dealing damage to all entities within range."""

    # Class variables
    attack_type: AttackAlgorithmType = AttackAlgorithmType.AREA_OF_EFFECT

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<AreaOfEffectAttack (Owner={self.owner})>"

    def process_attack(self, *args: Any) -> None:
        """Perform an area of effect attack around the entity.

        Parameters
        ----------
        args: Any
            A tuple containing the parameters needed for the attack.
        """
        # Make sure the needed parameters are valid
        target_entity: arcade.SpriteList | Entity = args[0]
        logger.info(
            "Entity %r is performing an area of effect attack on %r",
            self.owner,
            target_entity,
        )

        # Create a sprite with an empty texture
        base_size = int(self.attack_data.attack_range * 2 * SPRITE_SIZE)
        empty_texture = arcade.Texture.create_empty(
            "",
            (base_size, base_size),
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
                target_entity.deal_damage(self.attack_data.damage)
        except TypeError:
            for entity in arcade.check_for_collision_with_list(
                area_of_effect_sprite, target_entity
            ):  # type: Entity
                # Deal damage to all the enemies within range
                entity.deal_damage(self.attack_data.damage)


ATTACKS = {
    AttackAlgorithmType.RANGED: RangedAttack,
    AttackAlgorithmType.MELEE: MeleeAttack,
    AttackAlgorithmType.AREA_OF_EFFECT: AreaOfEffectAttack,
}


def create_attack(
    owner: Entity, attack_type: AttackAlgorithmType, attack_data: AttackData
) -> AttackBase:
    """Determine which attack algorithm should be created based on a given attack type.

    Parameters
    ----------
    owner: Entity
        The reference to the entity object.
    attack_type: AttackAlgorithmType
        The attack algorithm to create.
    attack_data: AttackData
        The attack data for this attack.
    """
    # Get the attack algorithm class type which matches the given attack type
    cls = ATTACKS[attack_type]
    logger.debug(
        "Selected attack algorithm %r for given attack algorithm type %r",
        cls,
        attack_type,
    )

    # Initialise the class with the given parameters
    return cls(owner, attack_data)
