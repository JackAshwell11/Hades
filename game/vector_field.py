"""Creates an efficient vector flow field useful for navigating enemies around the game
map."""
from __future__ import annotations

# Builtin
import logging
import time
from collections import deque
from typing import TYPE_CHECKING

# Pip
import numpy as np

# Custom
from game.constants.game_object import SPRITE_SIZE

if TYPE_CHECKING:
    import arcade

__all__ = ("VectorField",)

# Get the logger
logger = logging.getLogger(__name__)


class VectorField:
    """Represents a vector flow field that allows for efficient pathfinding to a
    specific position for large amount of entities.

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
        will follow. Repeating this for every tile in the flow file gives us an
        efficient way to calculate pathfinding for a large amount of entities.

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
        return f"<VectorField (Width={self.width}) (Height={self.height})"

    def _get_neighbours(
        self, tile_pos: tuple[int, int], offsets: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Gets a tile position's floor neighbours based on a given list of offsets.

        Parameters
        ----------
        tile_pos: tuple[int, int]
            The tile position to get neighbours for.
        offsets: list[tuple[int, int]]
            A list of offsets used for getting the tile position's neighbours.

        Returns
        -------
        list[tuple[int, int]]
            A list of the tile position's neighbours.
        """
        # Get all the neighbour floor tile positions relative to the current tile
        # position
        tile_neighbours: list[tuple[int, int]] = []
        for dx, dy in offsets:
            # Check if the neighbour position is within the boundaries or not
            x, y = tile_pos[0] + dx, tile_pos[1] + dy
            if (x < 0 or x >= self.width) or (y < 0 or y >= self.height):
                continue

            # Check if the neighbour is a wall or not
            if self.distances.get((x, y), -1) == np.inf:
                continue

            # Neighbour tile position is a floor tile position, so it is valid
            tile_neighbours.append((x, y))

        # Return all the neighbours
        return tile_neighbours

    @staticmethod
    def get_tile_pos_for_pixel(position: tuple[float, float]) -> tuple[int, int]:
        """Converts a sprite position on the screen into a tile position for use with
        the vector field.

        Parameters
        ----------
        position: tuple[float, float]
            The sprite position on the screen.

        Returns
        -------
        tuple[int, int]
            The path field grid tile position for the given sprite position.
        """
        return int(position[0] // SPRITE_SIZE), int(position[1] // SPRITE_SIZE)

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
            if not current:
                continue

            # Get the current tile's neighbours
            for neighbour in self._get_neighbours(current, self._no_diagonal_offsets):
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
            for neighbour in self._get_neighbours(tile, self._diagonal_offsets):
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
        """Gets the vector the enemy needs to travel in based on their position.

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
