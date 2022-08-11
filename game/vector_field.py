"""Creates a vector field useful for navigating enemies around the game map."""
from __future__ import annotations

# Builtin
import logging
import time
from collections import deque
from typing import TYPE_CHECKING

# Pip
import numpy as np

# Custom
from game.common import grid_bfs
from game.constants.game_object import SPRITE_SIZE

if TYPE_CHECKING:
    import arcade

__all__ = ("VectorField",)

# Get the logger
logger = logging.getLogger(__name__)


intercardinal_offsets: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


class VectorField:
    """Represents a vector field allowing pathfinding for large amounts of enemies.

    The steps needed to accomplish this:
        1. First, we start at the destination tile and work our way outwards using a
        breadth first search. This is called a 'flood fill' and will construct the
        Dijkstra map needed for the vector field.

        2. Next, we iterate over each tile and find the neighbour with the lowest
        Dijkstra distance. Using this we can create a vector from the source tile to the
        neighbour tile making for more natural pathfinding since the enemy can go in 6
        directions instead of 4.

        3. Finally, once the neighbour with the lowest Dijkstra distance is found, we
        can create a vector from the current tile to that neighbour tile which the enemy
        will follow. Repeating this for every tile gives us an efficient way to
        calculate pathfinding for a large amount of entities.

    Further reading which may be useful:
    `Other uses of Dijkstra maps
    <http://www.roguebasin.com/index.php/The_Incredible_Power_of_Dijkstra_Maps>`_
    `Dijkstra maps visualized
    <http://www.roguebasin.com/index.php/Dijkstra_Maps_Visualized>`_
    `Understanding goal based pathfinding
    <https://gamedevelopment.tutsplus.com/tutorials/understanding-goal-based-vector\
    -field-pathfinding--gamedev-9007>`_

    Parameters
    ----------
    width: int
        The width of the grid.
    height: int
        The height of the grid.
    walls: arcade.SpriteList
        A list of wall sprites that can block the entities.

    Attributes
    ----------
    walls_dict: dict[tuple[int, int], int]
        A dictionary which persistently holds the wall tile positions and their
        distances from the destination tile. This is used to update the main distances
        dict when the vector field is recalculated.
    distances: dict[tuple[int, int], int]
        A dictionary which holds a tuple containing the tile positions and its distance
        from the destination tile.
    vector_dict: dict[tuple[int, int], tuple[float, float]]
        A dictionary which holds a tuple containing the tile position and the vector the
        enemy should move on that when on that tile.
    """

    __slots__ = (
        "width",
        "height",
        "walls_dict",
        "distances",
        "vector_dict",
    )

    def __init__(
        self,
        width: int,
        height: int,
        walls: arcade.SpriteList,
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.walls_dict: dict[tuple[int, int], int] = {}
        self.distances: dict[tuple[int, int], int] = {}
        self.vector_dict: dict[tuple[int, int], tuple[float, float]] = {}
        for wall in walls:
            self.walls_dict[self.get_tile_pos_for_pixel(wall.position)] = np.inf

    def __repr__(self) -> str:
        """Return a human-readable representation of this object."""
        return f"<VectorField (Width={self.width}) (Height={self.height})>"

    @staticmethod
    def get_tile_pos_for_pixel(position: tuple[float, float]) -> tuple[int, int]:
        """Calculate the tile position from a given screen position.

        Parameters
        ----------
        position: tuple[float, float]
            The sprite position on the screen.

        Raises
        ------
        ValueError
            The inputs must be bigger than or equal to 0.

        Returns
        -------
        tuple[int, int]
            The path field grid tile position for the given sprite position.
        """
        # Check if the inputs are negative
        x, y = position
        if x < 0 or y < 0:
            raise ValueError("The inputs must be bigger than or equal to 0.")

        # Calculate the grid position
        return int(x // SPRITE_SIZE), int(y // SPRITE_SIZE)

    def recalculate_map(self, player_pos: tuple[float, float]) -> None:
        """Recalculates the vector field and produces a new path_dict.

        Parameters
        ----------
        player_pos: tuple[float, float]
            The position of the player on the screen.
        """
        # Record the start time, so we can know how long the generation takes
        start_time = time.perf_counter()

        # To recalculate the map, we need a few things:
        #   1. A distances dict to store the distances to each tile position from the
        #   destination tile position. This needs to only include the elements inside
        #   the walls dict.
        #   2. A vector_dict dict to store the paths for the vector field. We also need
        #   to make sure this is empty first.
        #   3. A queue object, so we can explore the grid.
        start = self.get_tile_pos_for_pixel(player_pos)
        self.distances.clear()
        self.distances |= self.walls_dict
        self.distances[start] = 0
        self.vector_dict.clear()
        queue: deque[tuple[int, int]] = deque[tuple[int, int]]()
        queue.append(start)

        # Explore the grid using a breadth first search to generate the Dijkstra
        # distances
        while bool(queue):
            # Get the current tile to explore
            current = queue.popleft()

            # Sometimes current can be None, so check if it is None
            if not current:  # pragma: no cover
                continue

            # Get the current tile's neighbours
            for neighbour in grid_bfs(current, self.height, self.width):
                # Check if the neighbour is a wall or not
                if self.distances.get((neighbour[0], neighbour[1]), -1) == np.inf:
                    continue

                # Test if the neighbour has already been reached or not. If it hasn't,
                # add it to the queue and set its distance
                if neighbour not in self.distances:
                    queue.append(neighbour)
                    self.distances[neighbour] = 1 + self.distances[current]

        # Use the newly generated Dijkstra map to get a neighbour with the lowest
        # Dijkstra distance at each tile. Then create a vector pointing in that
        # direction
        for tile, cost in self.distances.items():
            # If this tile is a wall tile, ignore it
            if cost == np.inf:
                continue

            # Find the tile's neighbour with the lowest Dijkstra distance
            min_tile = -1, -1
            min_dist = np.inf
            for neighbour in grid_bfs(
                tile, self.width, self.height, offsets=intercardinal_offsets
            ):
                # Sometimes an invalid tile is returned so test for that
                distance = self.distances.get(neighbour, -1)
                if distance < min_dist and distance != -1:
                    min_tile = neighbour
                    min_dist = distance

            # Now point the tile's vector in the direction of the tile with the lowest
            # Dijkstra distance
            self.vector_dict[tile] = -(tile[0] - min_tile[0]), -(tile[1] - min_tile[1])

        # Set the vector for the destination tile to avoid weird movement when the enemy
        # is touching the player
        self.vector_dict[start] = 0, 0

        # Log the time taken to generate the vector field
        logger.debug(
            "Vector field generated in %f seconds", time.perf_counter() - start_time
        )

    def get_vector_direction(
        self, current_enemy_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """Get the vector the enemy should travel on based on their current position.

        Parameters
        ----------
        current_enemy_pos: tuple[float, float]
            The current position of the enemy on the screen.

        Returns
        -------
        tuple[float, float[
            The vector the enemy needs to travel in.
        """
        return self.vector_dict[self.get_tile_pos_for_pixel(current_enemy_pos)]
