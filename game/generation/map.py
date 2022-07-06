"""Manages the procedural generation of the dungeon and places the player, enemies and
items into the game map."""
from __future__ import annotations

# Builtin
import logging
from collections import deque
from itertools import pairwise
from typing import NamedTuple

# Pip
import numpy as np

# Custom
from game.constants.constructor import ENEMIES
from game.constants.general import DEBUG_LINES
from game.constants.generation import (
    BASE_ENEMY_COUNT,
    BASE_ITEM_COUNT,
    BASE_MAP_HEIGHT,
    BASE_MAP_WIDTH,
    BASE_OBSTACLE_COUNT,
    BASE_SPLIT_ITERATION,
    ENEMY_DISTRIBUTION,
    HALLWAY_SIZE,
    ITEM_DISTRIBUTION,
    MAX_ENEMY_COUNT,
    MAX_ITEM_COUNT,
    MAX_MAP_HEIGHT,
    MAX_MAP_WIDTH,
    MAX_OBSTACLE_COUNT,
    MAX_SPLIT_ITERATION,
    PLACE_TRIES,
    SAFE_SPAWN_RADIUS,
    TileType,
)
from game.generation.astar import calculate_astar_path
from game.generation.bsp import Leaf
from game.generation.primitives import Point, Rect

__all__ = (
    "GameMapShape",
    "Map",
    "create_map",
)

# Get the logger
logger = logging.getLogger(__name__)

# Set the numpy print formatting to allow pretty printing (for debugging)
np.set_printoptions(threshold=1, edgeitems=50, linewidth=10000)


def create_map(level: int) -> tuple[np.ndarray, GameMapShape]:
    """Initialises and generates the game map.

    Parameters
    ----------
    level: int
        The game level to generate a map for.

    Returns
    -------
    tuple[np.ndarray, GameMapShape]
        The generated map and a named tuple containing the width and height.
    """
    grid: np.ndarray = Map(level).grid
    return grid, GameMapShape(grid.shape[1], grid.shape[0])


class GameMapShape(NamedTuple):
    """Represents a two element tuple holding the width and height of a game map.

    width: int
        The width of the game map.
    height: int
        The height of the game map.
    """

    width: int
    height: int


class Map:
    """Procedurally generates a dungeon based on a given game level.

    Parameters
    ----------
    level: int
        The game level to generate a map for.

    Attributes
    ----------
    map_constants: dict[TileType | str, int]
        A mapping of constant name to value. These constants are width, height, split
        count (how many times the bsp should split) and the counts for the different
        enemies and items.
    grid: np.ndarray
        The 2D grid which represents the dungeon.
    bsp: Leaf
        The root leaf for the binary space partition.
     player_pos: tuple[int, int]
        The player's position in the grid. This is set to (-1, -1) to avoid typing
        errors.
    enemy_spawns: list[tuple[int, int]]
        The coordinates for the enemy spawn points. This is in the format (x, y).
    """

    __slots__ = (
        "level",
        "map_constants",
        "grid",
        "bsp",
        "player_pos",
        "enemy_spawns",
    )

    def __init__(self, level: int) -> None:
        self.level: int = level
        self.map_constants: dict[TileType | str, int] = self._generate_constants()
        self.grid: np.ndarray = np.full(
            (self.map_constants["height"], self.map_constants["width"]), 0, np.int8
        )
        self.bsp: Leaf = Leaf(
            Point(0, 0),
            Point(self.map_constants["width"] - 1, self.map_constants["height"] - 1),
            None,
            self.grid,
        )
        self.player_pos: tuple[int, int] = (-1, -1)
        self.enemy_spawns: list[tuple[int, int]] = []

        # Create the map
        self._split_bsp()
        self._create_hallways(self._generate_rooms())
        print(self.grid)

    def __repr__(self) -> str:
        return (
            f"<Map (Width={self.map_constants['width']})"
            f" (Height={self.map_constants['height']}) (Split"
            f" count={self.map_constants['split count']}) (Enemy"
            f" count={self.map_constants['enemy count']}) (Item"
            f" count={self.map_constants['item count']})>"
        )

    @property
    def width(self) -> int:
        """Gets the width of the grid.

        Returns
        -------
        int
            The width of the grid.
        """
        # Make sure variables needed are valid
        assert self.grid is not None

        # Return the shape
        return self.grid.shape[1]

    @property
    def height(self) -> int:
        """Gets the height of the grid.

        Returns
        -------
        int
            The height of the grid.
        """
        # Make sure variables needed are valid
        assert self.grid is not None

        # Return the shape
        return self.grid.shape[0]

    def _generate_constants(self) -> dict[TileType | str, int]:
        """Generates the needed constants based on a given level.

        Returns
        -------
        dict[TileType | str, int]
            The generated constants.
        """
        # Create the generation constants
        generation_constants: dict[TileType | str, int] = {
            "width": np.minimum(
                int(np.round(BASE_MAP_WIDTH * 1.2**self.level)), MAX_MAP_WIDTH
            ),
            "height": np.minimum(
                int(np.round(BASE_MAP_HEIGHT * 1.2**self.level)), MAX_MAP_HEIGHT
            ),
            "split iteration": np.minimum(
                int(np.round(BASE_SPLIT_ITERATION * 1.5**self.level)),
                MAX_SPLIT_ITERATION,
            ),
            "obstacle count": np.minimum(
                int(np.round(BASE_OBSTACLE_COUNT * 1.3**self.level)),
                MAX_OBSTACLE_COUNT,
            ),
            "enemy count": np.minimum(
                int(np.round(BASE_ENEMY_COUNT * 1.1**self.level)), MAX_ENEMY_COUNT
            ),
            "item count": np.minimum(
                int(np.round(BASE_ITEM_COUNT * 1.1**self.level)), MAX_ITEM_COUNT
            ),
        }

        # Create the dictionary which will hold the counts for each enemy and item type
        type_dict: dict[TileType, int] = {
            key: int(np.ceil(value * generation_constants["enemy count"]))
            for key, value in ENEMY_DISTRIBUTION.items()
        } | {
            key: int(np.ceil(value * generation_constants["item count"]))
            for key, value in ITEM_DISTRIBUTION.items()
        }

        # Merge the enemy/item type count dict and the generation constants dict
        # together and then return the result
        result = generation_constants | type_dict
        logger.info("Generated map constants %r", result)
        return result

    def _split_bsp(self) -> None:
        """Splits the bsp based on the generated constants."""
        # Start the splitting using deque
        deque_obj = deque["Leaf"]()
        deque_obj.append(self.bsp)
        split_iteration = self.map_constants["split iteration"]
        while split_iteration and deque_obj:
            # Get the current leaf from the deque object
            current = deque_obj.popleft()

            # Split the bsp if possible
            if current.split(DEBUG_LINES) and current.left and current.right:
                # Add the child leafs so they can be split
                logger.debug("Split bsp. Split iteration is now %d", split_iteration)
                deque_obj.append(current.left)
                deque_obj.append(current.right)

                # Decrement the split count
                split_iteration -= 1

    def _generate_rooms(self) -> list[Rect]:
        """Generates the rooms for a given level using the bsp.

        Returns
        -------
        list[Rect]
            The generated rooms.
        """
        # Create the rooms. We can use the same deque object since it is currently empty
        rooms: list[Rect] = []
        deque_obj = deque["Leaf"]()
        deque_obj.append(self.bsp)
        while deque_obj:
            # Get the current leaf from the stack
            current = deque_obj.pop()

            # Check if a room already exists in this leaf
            if current.room:
                continue

            # Test if we can create a room in the current leaf
            if current.left and current.right:
                # Room creation not successful meaning there are child leafs so try
                # again on the child leafs
                deque_obj.append(current.left)
                deque_obj.append(current.right)
            else:
                # Create a room in the current leaf and save the rect
                logger.debug("Creating room in leaf %r", current)
                while not current.create_room():
                    # Width to height ratio is outside of range so try again
                    logger.debug("Trying generation of room in leaf %r again", current)

                # Check if the room was actually created. If so, append it to the list
                if current.room:
                    rooms.append(current.room)

        # Return all the created rooms
        return rooms

        # import matplotlib.pyplot as plt
        # import networkx as nx
        #
        # g = nx.DiGraph()
        # i = 0
        # y = {}
        # deque_obj.append(self.bsp)
        # while deque_obj:
        #     current = deque_obj.pop()
        #     current_dict = y.get(current, -1)
        #     if current_dict == -1:
        #         i += 1
        #         y[current] = i
        #         current_dict = i
        #
        #     if current.left:
        #         current_left_dict = y.get(current.left, -1)
        #         if current_left_dict == -1:
        #             i += 1
        #             y[current.left] = i
        #             current_left_dict = i
        #         g.add_edge(current_dict, current_left_dict)
        #         deque_obj.append(current.left)
        #     if current.right:
        #         current_right_dict = y.get(current.right, -1)
        #         if current_right_dict == -1:
        #             i += 1
        #             y[current.right] = i
        #             current_right_dict = i
        #         g.add_edge(current_dict, current_right_dict)
        #         deque_obj.append(current.right)
        # nx.draw_planar(g, arrows=True, with_labels=True)
        # plt.show()

    def _create_hallways(self, rooms: list[Rect]):
        """Creates the hallways by generating a Delaunay graph and finding a minimum
        spawning tree before adding a few removed edges back into the graph. This
        ensures that the hallways won't intersect and that there won't be too many.

        Parameters
        ----------
        rooms: list[Rects]
            The rooms to create a Delaunay graph out of.
        """
        # Place random obstacles in the grid
        y, x = np.where(self.grid == 0)
        arr_index = np.random.choice(len(y), self.map_constants["obstacle count"])
        self.grid[y[arr_index], x[arr_index]] = TileType.OBSTACLE

        # # Create a complete graph out of rooms
        # connections: set[tuple[Point, Point, float]] = set()
        # complete_graph: dict[Point, list[tuple[Point, float]]] = {}
        # from itertools import permutations
        # from heapq import heappop, heappush
        # for source, destination in permutations(rooms, 2):
        #     cost = source.get_distance_to(destination)
        #     source_center = source.center
        #     destination_center = destination.center
        #     complete_graph.update(
        #         {
        #             source_center: complete_graph.get(source_center, [])
        #             + [(destination_center, cost)]
        #         }
        #     )
        #     connections.add((source_center, destination_center, cost))
        #
        # # Use Prim's algorithm to construct a minimum spanning tree of complete_graph
        # visited: set[Point] = set()
        # start = next(iter(complete_graph))
        # unexplored: list[tuple[float, Point, Point]] = [(0, start, start)]
        # mst: set[tuple[Point, Point, float]] = set()
        # while unexplored:
        #     # Get the neighbour with the lowest cost
        #     cost, source, destination = heappop(unexplored)
        #
        #     # Check if the neighbour is already visited or not
        #     if destination not in visited:
        #         # Neighbour isn't visited so mark them as visited and add their
        #         # neighbours to the heap
        #         visited.add(destination)
        #         for neighbour, neighbour_cost in complete_graph[destination]:
        #             if neighbour not in visited:
        #                 heappush(unexplored, (neighbour_cost, destination, neighbour))
        #
        #         # Add a new edge towards the lowest cost neighbour onto the mst
        #         if source != destination:
        #             mst.add((source, destination, cost))
        #
        # # Add some removed edges back into the graph, so it's not as sparsely populated
        # removed_edges = connections - connections.intersection(mst)
        # hallway_connections = mst.copy().union(
        #     {
        #         removed_edges.pop()
        #         for _ in range(round(len(removed_edges) * 0.15))
        #     }
        # )

        # Use the A* algorithm with to connect each pair of rooms making sure to avoid
        # the obstacles giving us natural looking hallways. Note that the width of the
        # hallways will always be odd in this implementation due to numpy indexing
        half_hallway_size = HALLWAY_SIZE // 2
        for pair_source, pair_destination in pairwise(rooms):
            for path_point in calculate_astar_path(
                self.grid,
                Point(*pair_source.center),
                Point(*pair_destination.center),
            ):
                # Test if the current tile is a floor tile
                if self.grid[path_point.y][path_point.x] is TileType.FLOOR:
                    # Current tile is a floor tile, so there is no point placing a rect
                    continue

                # Place a rect box around the path_point using HALLWAY_SIZE to determine
                # the width and height
                Rect(
                    self.grid,
                    Point(
                        path_point.x - half_hallway_size,
                        path_point.y - half_hallway_size,
                    ),
                    Point(
                        path_point.x + half_hallway_size,
                        path_point.y + half_hallway_size,
                    ),
                ).place_rect()

    def f(self):
        # from scipy.spatial import Delaunay
        # import matplotlib.pyplot as plt
        # points = []
        # for leaf in leafs:
        #     points.append([leaf.room.center_x, leaf.room.center_y])
        # points = np.array(points)
        # tri = Delaunay(points)
        # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
        # plt.plot(points[:, 0], points[:, 1], 'o')
        # plt.show()

        # Get all the rooms objects from the leafs list, so we can store the hallways
        # too. To make the hallways, we can connect each pair of leaves in the leafs
        # list using itertools.pairwise
        rooms: list[Rect] = [leaf.room for leaf in leafs if leaf.room]
        hallways: list[Rect] = []
        logger.info("Created %d rooms", len(rooms))
        for pair in list(pairwise(leafs)):
            first_hallway, second_hallway = pair[0].create_hallway(pair[1])
            if first_hallway:
                hallways.append(first_hallway)
            if second_hallway:
                hallways.append(second_hallway)
        logger.info("Created %d hallways", len(hallways))

        # Create a sorted list of tuples based on the rect areas
        rects: list[Rect] = rooms + hallways
        rect_areas = sorted(
            ((rect, rect.width * rect.height) for rect in rects),
            key=lambda x: x[1],
        )
        total_area = sum(area[1] for area in rect_areas)
        logger.debug("Created %d total rects with area %d", len(rects), total_area)

        # Place the player spawn in the smallest room
        self._place_tile(TileType.PLAYER, rect_areas[0][0])

        # Place the enemies
        self._place_enemies(
            rect_areas,
            [area[1] / total_area for area in rect_areas],
        )

        # Place the items
        self._place_items(rect_areas)
        logger.info(
            "Finished creating game map with constants %r and rect count %d",
            self.map_constants,
            len(rects),
        )

    def _place_enemies(
        self,
        rect_areas: list[tuple[Rect, int]],
        area_probabilities: list[float],
    ) -> None:
        """Places the enemies in the grid making sure other tiles aren't replaced.

        Parameters
        ----------
        rect_areas: list[tuple[Rect, int]]
            A sorted list of rects and their areas.
        area_probabilities: list[float]
            A list of areas probabilities. This corresponds to rect_areas.
        """
        # Repeatedly place an enemy type. If they are placed, we can increment the
        # counter. Otherwise, continue
        for enemy in ENEMY_DISTRIBUTION:
            # Set up the counters for this enemy type
            count = self.map_constants[enemy]
            enemies_placed = 0
            tries = PLACE_TRIES
            while enemies_placed < count and tries != 0:
                if self._place_tile(
                    enemy,
                    np.random.choice(
                        [rect[0] for rect in rect_areas], p=area_probabilities
                    ),
                ):
                    # Enemy placed
                    enemies_placed += 1
                else:
                    # Enemy not placed
                    tries -= 1

    def _place_items(self, rect_areas: list[tuple[Rect, int]]) -> None:
        """Places the items in the grid making sure other tiles aren't replaced.

        Parameters
        ----------
        rect_areas: list[tuple[Rect, int]]
            A sorted list of rects and their areas. This is only used to pick a random
            rect, the items aren't actually placed based on weights.
        """
        # Repeatedly place an item type. If they are placed, we can increment the
        # counter. Otherwise, continue
        for item in ITEM_DISTRIBUTION:
            # Set up the counters for this item type
            count = self.map_constants[item]
            items_placed = 0
            tries = PLACE_TRIES
            while items_placed < count and tries != 0:
                if self._place_tile(
                    item, np.random.choice([rect[0] for rect in rect_areas])
                ):
                    # Item placed
                    items_placed += 1
                else:
                    # Item not placed
                    tries -= 1

    def _place_tile(self, entity: TileType, rect: Rect) -> bool:
        """Places a given entity in a random position in a given rect.

        Parameters
        ----------
        entity: TileType
            The entity to place in the grid.
        rect: Rect
            The rect object to place the tile in.

        Returns
        -------
        bool
            Whether or not an enemy was placed.
        """
        # Make sure variables needed are valid
        assert self.grid is not None

        # Get a random position within the rect making sure to exclude the walls
        position_x, position_y = (
            np.random.randint(rect.top_left.x + 1, rect.bottom_right.x - 1),
            np.random.randint(rect.top_left.y + 1, rect.bottom_right.y - 1),
        )

        # Check if the entity is an enemy. If so, we need to make sure they are not
        # within the spawn radius
        if entity in ENEMIES:
            distance_to_player = np.hypot(
                self.player_pos[0] - position_x, self.player_pos[1] - position_y
            )
            if distance_to_player < SAFE_SPAWN_RADIUS:
                # Enemy is within spawn radius so don't place them
                return False

        # Check if the chosen position is already taken
        if self.grid[position_y, position_x] != TileType.FLOOR.value:
            # Already taken
            return False

        # Place the entity in the random position
        self.grid[position_y, position_x] = entity.value

        # Check if the entity is the player. If so, save the position
        if entity is TileType.PLAYER:
            self.player_pos = (position_x, position_y)

        # Return true so we know an enemy has been placed
        return True
