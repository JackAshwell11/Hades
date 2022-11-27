"""Holds stub data for astar.cpp to help with type inference."""
from __future__ import annotations

# Pip
import numpy as np
import numpy.typing as npt

# Custom
from hades.generation.primitives import Point

def calculate_astar_path(
    grid: npt.NDArray[np.int8], start: Point, end: Point
) -> list[tuple[int, int]]: ...
