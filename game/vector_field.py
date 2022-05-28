from __future__ import annotations

# Pip
import numpy as np


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
        "draw_distances",
    )

    def __init__(self, game_map: np.ndarray, draw_distances: bool = False) -> None:
        self.game_map: np.ndarray = game_map
        self.draw_distances: bool = draw_distances
