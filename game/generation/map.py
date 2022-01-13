from __future__ import annotations

# Builtin
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Pip
import numpy as np

# Custom
from .rooms import (
    RectInstance,
    Template,
    large_room,
    medium_room,
    small_room,
    starting_room,
)

# Constants
EMPTY = 0
FLOOR = 1
WALL = 2
PLAYER_START = 3

MAX_WIDTH = 100
MAX_HEIGHT = 60
MAX_RECT_COUNT = 20


@dataclass()
class Rect:
    """
    Represents a rectangle in the game of any size.

    Parameters
    ----------
    x: int
        The x position of the top-left corner.
    y: int
        The y position of the top-left corner
    width: int
        The width of the rectangle.
    height: int
        The height of the rectangle.
    rect_type: Template
        The type of rect the current object is.
    """

    x: int
    y: int
    width: int
    height: int
    rect_type: Template

    @property
    def bottom_right(self) -> Tuple[int, int]:
        """Returns the x, y position of the bottom right corner of the rect."""
        return self.x + self.width - 1, self.y + self.height - 1

    @property
    def center(self) -> Tuple[int, int]:
        """Returns the center position of the rect."""
        return (
            int((self.x + self.bottom_right[0]) / 2),
            int((self.y + self.bottom_right[1]) / 2),
        )

    def intersect(self, other: Rect) -> bool:
        """
        Checks if this rect object intersects with another rect object.

        Parameters
        ----------
        other: Rect
            The rect object to check for an intersection.

        Returns
        -------
        bool
            Whether or not the two rect objects intersect.
        """
        return not (
            # These checks test if the top left or bottom right corners of each object
            # are past each other effectively meaning they cannot intersect
            self.x > other.bottom_right[0]
            or self.bottom_right[0] < other.x
            or self.y > other.bottom_right[1]
            or self.bottom_right[1] < other.y
        )


class Map:
    """
    Procedurally generates a game generation based on a given game level.

    Parameters
    ----------
    level: int
        The game level to generate a map for.

    Attributes
    ----------
    width: int
        The width of the game generation.
    height: int
        The height of the game generation.
    max_level_rect_count: int
        The maximum number of rects allowed in this level.
    grid: Optional[np.ndarray]
        The actual 2D matrix which represents the map.
    rooms: List[Rect]
        A list of rect object which have been generated.
    last_calculation: Dict[str, float]
        The probabilities calculated after the current rect has been spawned. These are
        calculated in the door object by multiplying the parent's last calculation by
        the probabilities for the new object which is to be spawned.
    """

    def __init__(self, level: int) -> None:
        self.level: int = level
        self.width: int = -1
        self.height: int = -1
        self.max_level_rect_count: int = -1
        self.grid: Optional[np.ndarray] = None
        self.rooms: List[Rect] = []
        self.last_calculation: Dict[str, float] = starting_room.probabilities
        self.templates: Dict[str, Template] = {
            RectInstance.SMALL_ROOM.value: small_room,
            RectInstance.MEDIUM_ROOM.value: medium_room,
            RectInstance.LARGE_ROOM.value: large_room,
        }
        self.make_map(self.level)

    def __repr__(self) -> str:
        return f"<Map (Width={self.width}) (Height={self.height})>"

    def make_map(self, level: int) -> None:
        """
        Function which actually does the map generation for a specified level.

        Parameters
        ----------
        level: int
            The level to create a game generation for.
        """

        # Create constants used during the generation
        # TO DO
        self.width = 30
        self.height = 20
        self.max_level_rect_count = 5

        # Create the 2D grid
        self.grid = np.full((self.height, self.width), EMPTY, np.int8)

        # Create starting room
        self.generate_start_room()
        # Generate the rects and hallways
        self.generate_rects()

        print(self.grid)

    def generate_start_room(self) -> None:
        """
        Creates a 5x5 starting room at a random position making sure it doesn't exceed
        the array bounds. Note that the int after the : is 1 bigger since numpy doesn't
        include the starting pos.

        This will create a 2D matrix at a random point that looks like:
        2  2  2  2  2  2  2
        2  1  1  1  1  1  2
        2  1  1  1  1  1  2
        2  1  1  3  1  1  2
        2  1  1  1  1  1  2
        2  1  1  1  1  1  2
        2  2  2  2  2  2  2
        """
        # Check the grid was created successfully
        assert isinstance(self.grid, np.ndarray)
        # Create starting position
        starting_pos_x, starting_pos_y = (
            random.randint(4, self.width - 4),
            random.randint(4, self.height - 4),
        )
        # Get top-left corner
        top_left_x, top_left_y = (starting_pos_x - 3, starting_pos_y - 3)
        # Create starting rect object
        start_rect = Rect(
            top_left_x,
            top_left_y,
            starting_room.min_width,
            starting_room.min_height,
            starting_room,
        )
        # Place the starting room
        self.create_room(start_rect)
        # Set player position
        self.grid[starting_pos_y, starting_pos_x] = PLAYER_START
        # Add the starting room rect to rooms
        self.rooms.append(start_rect)

    def generate_rects(self) -> None:
        """
        Tries to procedurally generate self.max_level_rect_count amount of rects at
        random positions in the game map then place hallways from the last generated
        rect to the newly generated one.
        """
        for count in range(self.max_level_rect_count):
            last_calculation = self.last_calculation
            # Pick the new rect
            result = random.choices(
                tuple(last_calculation.keys()),
                weights=tuple(last_calculation.values()),
                k=1,
            )[0]
            new_rect = self.templates[result]
            # Pick a random width, height and position
            width, height = (
                random.randint(new_rect.min_width, new_rect.max_width),
                random.randint(new_rect.max_width, new_rect.max_height),
            )
            x, y = (
                random.randint(0, self.width - width - 1),
                random.randint(0, self.height - height - 1),
            )
            # Create the rect object and check for collisions
            rect = Rect(x, y, width, height, new_rect)
            for other_room in self.rooms:
                if rect.intersect(other_room):
                    # Rect collides with other rects so its invalid
                    print("fail")
                    break
            else:
                # Rect is valid so recalculate the probabilities
                new_probabilities = self.process_calculations(
                    last_calculation, new_rect
                )
                # Place the rect in the game map
                self.create_room(rect)
                # Place the hallways
                previous_center_x, previous_center_y = self.rooms[-1].center
                new_center_x, new_center_y = rect.center
                # REMOVE FROM HERE
                self.create_horizontal_hallway(
                    previous_center_x, previous_center_y, new_center_x
                )
                self.create_vertical_hallway(
                    new_center_x, previous_center_y, new_center_y
                )
                print(self.grid)
                print("****")
                input()
                # REMOVE TO HERE
                if random.randint(0, 1) == 1:
                    # Move horizontally then vertically
                    self.create_horizontal_hallway(
                        previous_center_x, previous_center_y, new_center_x
                    )
                    self.create_vertical_hallway(
                        new_center_x, previous_center_y, new_center_y
                    )
                else:
                    # Move vertically then horizontally
                    self.create_vertical_hallway(
                        previous_center_x, previous_center_y, new_center_y
                    )
                    self.create_horizontal_hallway(
                        previous_center_x, new_center_y, new_center_x
                    )
                # Add the new rect to rooms and update the probabilities
                self.rooms.append(rect)
                self.last_calculation = new_probabilities

    @staticmethod
    def process_calculations(
        previous_calculation: Dict[str, float], new_rect: Template
    ) -> Dict[str, float]:
        """
        Recalculates the probabilities by multiplying the last calculation by the
        new_rect's probabilities then normalises the result.

        Parameters
        ----------
        previous_calculation: Dict[str, float]
            The previous calculation which is to be multiplied and replaced.
        new_rect: Template
            The rect to get the probabilities to multiply the previous calculation by.

        Returns
        -------
        Dict[str, float]
            The new probabilities
        """
        # Process calculations and normalise the result so all probabilities add up to
        # 1. This makes the numbers easier to work with
        assert new_rect is not None
        new_probabilities = {}
        for key, value in previous_calculation.items():
            new_probabilities[key] = value * new_rect.probabilities[key]
        return {
            key: value / sum(new_probabilities.values())
            for key, value in new_probabilities.items()
        }

    def create_room(self, rect: Rect) -> None:
        """
        Places a rect in the 2D grid.

        Parameters
        ----------
        rect: Rect
            The rect object to place in the 2D grid.
        """
        assert isinstance(self.grid, np.ndarray)
        # Create walls
        self.grid[
            rect.y : rect.y + rect.height,
            rect.x : rect.x + rect.width,
        ] = WALL
        # Create inner floors
        self.grid[
            rect.y + 1 : rect.y + rect.height - 1,
            rect.x + 1 : rect.x + rect.width - 1,
        ] = FLOOR

    def create_horizontal_hallway(
        self, previous_x: int, previous_y: int, new_x: int
    ) -> None:
        """
        Creates a horizontal hallway from one point to another.

        Parameters
        ----------
        previous_x: int
            The starting x point.
        previous_y: int
            The starting y point.
        new_x: int
            The ending x point.
        """
        assert isinstance(self.grid, np.ndarray)
        reach_empty = False
        for x in range(min(previous_x, new_x) - 1, max(previous_x, new_x) + 1):
            if self.grid[previous_y, x] == 0:
                # We've left the source room so stop at the next 1
                reach_empty = True
            if self.grid[previous_y, x] == 1 and reach_empty:
                # We've hit the next room so don't overwrite the floors
                break
            if self.grid[previous_y, x] == 3:
                # Don't overwrite the player spawn
                continue
            else:
                # We're free to change the values here. But walls should only replace
                # values of 0 otherwise the game map will become ugly
                # self.grid[previous_y - 2 : previous_y + 3, x] = WALL
                self.grid[previous_y - 1 : previous_y + 2, x] = FLOOR

    def create_vertical_hallway(
        self, previous_x: int, previous_y: int, new_y: int
    ) -> None:
        """
        Creates a vertical hallway from one point to another.

        Parameters
        ----------
        previous_x: int
            The starting x point.
        previous_y: int
            The starting y point.
        new_y: int
            The ending y point.
        """
        assert isinstance(self.grid, np.ndarray)
        reach_empty = False
        for y in range(min(previous_y, new_y) - 1, max(previous_y, new_y) + 1):
            if self.grid[y, previous_x] == 0:
                # We've left the source room so stop at the next 1
                reach_empty = True
            if self.grid[y, previous_x] == 1 and reach_empty:
                # We've hit the next room so don't overwrite the floors
                break
            if self.grid[y, previous_x] == 3:
                # Don't overwrite the player spawn
                continue
            else:
                # We're free to change the values here. But walls should only replace
                # values of 0 otherwise the game map will become ugly
                # self.grid[y, previous_x - 2 : previous_x + 3] = WALL
                self.grid[y, previous_x - 1 : previous_x + 2] = FLOOR
