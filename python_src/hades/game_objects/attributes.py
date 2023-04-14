"""Manages the different attributes available."""
from __future__ import annotations


__all__ = ()


class Health:
    maximum: bool = True
    status_effect: bool = False
    variable: bool = False

    def __init__(self: Health, **kwargs) -> None:
        self.health: int = 0
        print(kwargs)


class Armour:
    maximum: bool = True
    status_effect: bool = False
    variable: bool = False

    def __init__(self: Armour, **kwargs) -> None:
        self.armour: int = 0
        print(kwargs)
