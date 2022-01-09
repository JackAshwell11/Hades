from __future__ import annotations


class Tile:
    """
    Represents a single tile in the game generation.

    Parameters
    ----------
    x: int
        The x coordinate of the rectangle. This uses Python's array indexing.
    y: int
        The y coordinate of the rectangle. This uses Python's array indexing.
    """

    TILE_ID = -1

    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y

    def __repr__(self) -> str:
        return f"<Tile (Position=({self.x}, {self.y}))>"


class EmptyTile(Tile):
    """Represents an empty tile in the game generation."""

    TILE_ID = 0

    def __repr__(self) -> str:
        return f"<EmptyTile (Position=({self.x}, {self.y}))>"


class WallTile(Tile):
    """Represents a wall tile in the game generation."""

    TILE_ID = 1

    def __repr__(self) -> str:
        return f"<WallTile (Position=({self.x}, {self.y}))>"


class FloorTile(Tile):
    """Represents a floor tile in the game generation."""

    TILE_ID = 2

    def __repr__(self) -> str:
        return f"<FloorTile (Position=({self.x}, {self.y}))>"
