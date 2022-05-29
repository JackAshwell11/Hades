from __future__ import annotations

# Builtin
from collections import deque
from typing import TYPE_CHECKING

# Pip
import numpy as np

if TYPE_CHECKING:
    from game.entities.base import Tile


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

    @property
    def empty(self) -> bool:
        """
        Checks if the queue is empty or not.

        Returns
        -------
        bool
            Whether the queue is empty or not.
        """
        return not bool(self._queue)

    def put(self, tile: Tile) -> None:
        """
        Adds a tile to the queue.

        Parameters
        ----------
        tile: Tile
            The tile to add to the queue
        """
        self._queue.append(tile)

    def get(self) -> Tile:
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
    do_diagonals: bool
        Whether to include the diagonals in the exploration or not.
    """

    __slots__ = (
        "vector_grid",
        "destination_tile",
        "_neighbor_offsets",
    )

    _no_diagonal_offsets: list[tuple[int, int]] = [
        (0, -1),
        (-1, 0),
        (1, 0),
        (0, 1),
    ]

    _diagonal_offsets: list[tuple[int, int]] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]

    def __init__(
        self,
        vector_grid: np.ndarray,
        destination_tile: tuple[int, int],
        do_diagonals: bool = False,
    ) -> None:
        self.vector_grid: np.ndarray = vector_grid
        self.destination_tile: tuple[int, int] = destination_tile
        self._neighbor_offsets: list[tuple[int, int]] = (
            self._diagonal_offsets if do_diagonals else self._no_diagonal_offsets
        )
        self.recalculate_map()

    def __repr__(self) -> str:
        return (
            f"<VectorField (Width={self.vector_grid.shape[1]})"
            f" (Height={self.vector_grid.shape[0]})"
            f" (Destination={self.destination_tile})>"
        )

    @property
    def width(self) -> int:
        """
        Gets the width of the vector grid.

        Returns
        -------
        int
            The width of the vector grid.
        """
        return self.vector_grid.shape[1]

    @property
    def height(self) -> int:
        """
        Gets the height of the vector grid.

        Returns
        -------
        int
            The height of the vector grid.
        """
        return self.vector_grid.shape[0]

    def _get_neighbors(self, tile: Tile) -> list[Tile]:
        """
        Gets a tile's neighbors.

        Parameters
        ----------
        tile: Tile
            The tile to get neighbors for.

        Returns
        -------
        list[Tile]
            A list of the tile's neighbors.
        """
        # Check if each neighbour offset is within the map boundaries. If so, add that
        # tile to the list
        tile_neighbors: list[Tile] = []
        for dx, dy in self._neighbor_offsets:
            x, y = tile.tile_pos[0] + dx, tile.tile_pos[1] + dy
            target_tile: Tile = self.vector_grid[y][x]
            if self._point_in_map(x, y) and not target_tile.blocking:
                tile_neighbors.append(target_tile)

        # Return all the neighbors
        return tile_neighbors

    def _point_in_map(self, x: int, y: int) -> bool:
        """
        Checks if a given point is within the boundaries of the map or not.

        Returns
        -------
        x: int
            The x coordinate.
        y: int
            The y coordinate.

        Returns
        -------
        bool
            Whether the point is within the boundaries of the map or not.
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def recalculate_map(self) -> None:
        """Recalculates the Dijkstra map and generates the vector field."""
        # Create a queue object, so we can explore the grid, a came_from dict to
        # store the paths for the vector field and a distances' dict to store the
        # distances to each tile from the destination
        start = self.vector_grid[self.destination_tile[1]][self.destination_tile[0]]
        queue = Queue()
        queue.put(start)
        came_from: dict[Tile, Tile | None] = {start: None}
        distances: dict[Tile, int] = {start: 0}

        # Explore the grid using a breadth first search
        while not queue.empty:
            # Get the current tile to explore
            current = queue.get()

            # Get the current tile's neighbors
            for neighbor in self._get_neighbors(current):
                # Test if the neighbour has already been reached or not. If it hasn't,
                # add it to the queue, mark it as reached and set its distance
                if neighbor not in came_from:
                    queue.put(neighbor)
                    came_from[neighbor] = current
                    distances[neighbor] = 1 + distances[current]

        # start_tile: Tile | None = None
        # for tile in came_from:
        #     if tile.tile_pos == (15, 12):
        #         start_tile = tile
        #         break
        #
        # print(self.destination_tile)
        # print(f"Distance to destination: {distances[start_tile]}")
        #
        # path: list[Tile] = [start_tile]
        # while came_from[start_tile] is not None:
        #     start_tile = came_from[start_tile]
        #     path.append(start_tile)
        # print(path)
