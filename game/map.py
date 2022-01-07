from __future__ import annotations

# Builtin
from typing import List


class Map:
    """
    Procedurally generates a 2D list of integers representing a game map based on a
    given game level.

    Parameters
    ----------
    width: int
        The width of the game map.
    height: int
        The height of the game map.
    """

    def __init__(self, width: int, height: int):
        self.width: int = width
        self.height: int = height
        self.grid: List[List[int]] = [
            [0 for _ in range(self.width)] for _ in range(self.height)
        ]

    def make_map(self, level: int):
        """
        Function which actually creates the game map for a specified level.

        Parameters
        ----------
        level: int
            The level to create a game map for.
        """
        pass
