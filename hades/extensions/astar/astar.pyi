"""Holds stub data for astar.cpp to help with type inference."""
from __future__ import annotations

# Pip
import numpy as np

# Custom
from hades.constants.generation import TileType
from hades.generation.primitives import Point

def calculate_astar_path(
    grid: np.ndarray, start: Point, end: Point, obstacle_id: TileType
) -> list[tuple[int, int]]: ...
