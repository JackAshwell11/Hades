from __future__ import annotations

# Pip
import numpy as np


class Point:
    """"""

    __slots__ = (
        "tile_pos",
        "parent",
    )

    def __init__(self, tile_pos: tuple[int, int], parent: Point | None) -> None:
        self.tile_pos: tuple[int, int] = tile_pos
        self.parent: Point | None = parent

    def __repr__(self) -> str:
        return f"<Point (Tile pos={self.tile_pos}) (Parent={self.parent})>"


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
    game_map: np.ndarray
        The generated game map used to create the vector field.
    draw_distances: bool
        Whether to draw the Dijkstra map distances or not.
    """

    __slots__ = (
        "game_map",
        "destination_tile",
        "draw_distances",
        "vector_field",
    )

    def __init__(
        self,
        game_map: np.ndarray,
        destination_tile: tuple[int, int],
        draw_distances: bool = False,
    ) -> None:
        self.game_map: np.ndarray = game_map
        self.destination_tile: tuple[int, int] = destination_tile
        self.draw_distances: bool = draw_distances
        self.vector_field: np.ndarray = np.empty(self.game_map.shape, dtype=Point)
        self.recalculate_map()

    def __repr__(self) -> str:
        return (
            "<VectorField (Width=-1) (Height=-1)"
            f" (Destination={self.destination_tile})>"
        )

    def recalculate_map(self) -> None:
        """"""
        print(self.vector_field)
        print(self.destination_tile)
