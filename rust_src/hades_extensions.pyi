"""Holds stub data for the rust extension to help with type inference."""
from __future__ import annotations

class Point:
    x: int
    y: int

    def __init__(self, x: int, y: int) -> None: ...
    def sum(self, other: Point) -> tuple[int, int]: ...
    def abs_diff(self, other: Point) -> tuple[int, int]: ...

class Rect:
    top_left: Point
    bottom_right: Point
    width: int
    height: int
    center: Point

    def __init__(self, top_left: Point, bottom_right: Point) -> None: ...
    def get_distance_to(self, other: Rect) -> int: ...
    def place_rect(self) -> None: ...

def calculate_astar_path(
    grid: list[list[int]], start: Point, end: Point, obstacle_id: int
) -> list[tuple[int, int]]: ...
