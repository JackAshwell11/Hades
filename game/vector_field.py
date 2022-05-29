from __future__ import annotations

# Builtin
from collections import deque

# Pip
import numpy as np


class Tile:
    """
    Represents a tile in the game map

    Parameters
    ----------
    tile_pos: tuple[int, int]
        The position of the tile in the game map.
    is_blocking: bool
        Whether the tile is blocking or not.
    parent: Tile | None
        The tile object that discovered this tile.
    """

    __slots__ = (
        "tile_pos",
        "is_blocking",
        "parent",
    )

    def __init__(self, tile_pos: tuple[int, int], is_blocking: bool = False) -> None:
        self.tile_pos: tuple[int, int] = tile_pos
        self.is_blocking: bool = is_blocking
        self.parent: Tile | None = None

    def __repr__(self) -> str:
        return (
            f"<Tile (Tile pos={self.tile_pos}) (Is blocking={self.is_blocking})"
            f" (Parent={self.parent})>"
        )


class Queue:
    """Provides an abstraction over the deque object making access much easier."""

    __slots__ = ("_queue",)

    def __init__(self) -> None:
        self._queue: deque[Tile] = deque["Tile"]()

    def __repr__(self) -> str:
        return f"<Queue (Size={self.size})>"

    @property
    def size(self) -> int:
        """
        Gets the size of the queue.

        Returns
        -------
        int
            The size of the queue
        """
        return len(self._queue)

    def add(self, tile: Tile) -> None:
        """
        Adds a tile to the queue.

        Parameters
        ----------
        tile: Tile
            The tile to add to the queue
        """
        self._queue.append(tile)

    def remove(self) -> Tile:
        """
        Removes a tile from the queue.

        Returns
        -------
        Tile
            The tile that was removed from the queue.
        """
        return self._queue.popleft()


class VectorField:
    """
    Represents a vector flow field (or optionally a Dijkstra map) that allows for
    efficient  pathfinding to a specific position for large amount of entities. To steps
    needed to accomplish this:
        1. First, we start at the destination tile and work our way outwards using a
        breadth first search. This is called a 'flood fill' and will construct the base
        for our paths.

        2. Next, we draw vectors from each tile in the grid to their parent tile (the
        tile which discovered them or reached them). This constructs the paths which the
        entities can take to get to the destination tile).

        3. Finally, we can optionally add a value to each tile which is the distance
        from the current tile to the destination tile and is just the sum of the number
        of vectors to get to the destination tile. This is called a Dijkstra map and
        allows us to calculate the cost for a particular path.

    Further reading which may be useful:
        `Other uses of Dijkstra maps
        <http://www.roguebasin.com/index.php/The_Incredible_Power_of_Dijkstra_Maps>`_
        `Dijkstra maps visualized
        <http://www.roguebasin.com/index.php/Dijkstra_Maps_Visualized>`_

    Parameters
    ----------
    vector_grid: np.ndarray
        The generated game map which has the vector tiles already initialised.
    destination_tile: tuple[int, int]
        The destination tile which every tile will point towards.
    draw_distances: bool
        Whether to draw the Dijkstra map distances or not.
    """

    __slots__ = (
        "vector_grid",
        "destination_tile",
        "draw_distances",
    )

    def __init__(
        self,
        vector_grid: np.ndarray,
        destination_tile: tuple[int, int],
        draw_distances: bool = False,
    ) -> None:
        self.vector_grid: np.ndarray = vector_grid
        self.destination_tile: tuple[int, int] = destination_tile
        self.draw_distances: bool = draw_distances
        self.recalculate_map()

    def __repr__(self) -> str:
        return (
            f"<VectorField (Width={self.vector_grid.shape[1]})"
            f" (Height={self.vector_grid.shape[0]})"
            f" (Destination={self.destination_tile})>"
        )

    def recalculate_map(self) -> None:
        """Recalculates the Dijkstra map and generates the vector field."""
        # Create a queue object so we can explore the grid
        queue = Queue()
        print(self.destination_tile)
        print(queue)
