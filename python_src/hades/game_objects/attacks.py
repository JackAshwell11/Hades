"""Manages the different attack algorithms available."""
from __future__ import annotations

__all__ = ("RangedAttack", "MeleeAttack", "AreaOfEffectAttack")


class AreaOfEffectAttack:
    """Allows a game object to perform an area of effect attack around them."""

    def do_area_of_effect_attack(self: AreaOfEffectAttack) -> None:
        """Perform an area of effect attack around the entity."""
        raise NotImplementedError

    def __repr__(self: AreaOfEffectAttack) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return "<AreaOfEffectAttack>"


class MeleeAttack:
    """Allows a game object to perform a melee attack to other game objects."""

    def do_melee_attack(self: MeleeAttack) -> None:
        """Perform a melee attack in the direction the entity is facing."""
        raise NotImplementedError

    def __repr__(self: MeleeAttack) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return "<MeleeAttack>"


class RangedAttack:
    """Allows a game object to perform a ranged attack in a specific direction."""

    def do_ranged_attack(self: RangedAttack) -> None:
        """Perform a ranged attack in the direction the entity is facing."""
        raise NotImplementedError

    def __repr__(self: RangedAttack) -> str:
        """Return a human-readable representation of this object.

        Returns
        -------
        str
            The human-readable representation of this object.
        """
        return "<RangedAttack>"
