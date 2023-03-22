"""Holds stub data for the C++ extensions to help with type inference."""
from __future__ import annotations

# Builtin
from enum import Enum, auto

class TileType(int, Enum):
    DebugWall = auto()
    Empty = auto()
    Floor = auto()
    Wall = auto()
    Obstacle = auto()
    Player = auto()
    HealthPotion = auto()
    ArmourPotion = auto()
    HealthBoostPotion = auto()
    ArmourBoostPotion = auto()
    SpeedBoostPotion = auto()
    FireRateBoostPotion = auto()

def create_map(
    level: int, seed: int | None = None
) -> tuple[list[list[int]], tuple[int, int, int]]: ...
