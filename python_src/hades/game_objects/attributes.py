"""Manages the different attributes available."""
from __future__ import annotations

# Builtin
from dataclasses import dataclass

__all__ = ("StatusEffect",)


@dataclass(slots=True)
class StatusEffect:
    value: float
    duration: float


class EntityAttribute:
    pass


class Health:
    pass


class Armour:
    pass
