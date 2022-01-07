from __future__ import annotations

# Builtin
from typing import Optional

# Pip
import arcade

# Custom
from map import Map

MAP_WIDTH = 50
MAP_HEIGHT = 20


class Game(arcade.Window):
    """Manages the game and its actions."""

    def __init__(self) -> None:
        super().__init__()
        self.grid: Optional[Map] = None
        self.setup_level(1)

    def setup_level(self, level: int):
        """
        Creates the game map for a specific level.

        Parameters
        ----------
        level: int
            The level to create a map for. Each level is more difficult than the
            previous.
        """
        self.grid = Map(MAP_WIDTH, MAP_HEIGHT)
        self.grid.make_map(level)


def main() -> None:
    """Initialises the game and runs it."""
    window = Game()
    window.run()


if __name__ == "__main__":
    main()
