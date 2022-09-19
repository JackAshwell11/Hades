"""Holds stub data for vector_field.cpp to help with type inference."""
from __future__ import annotations

# Pip
import arcade

class VectorField:
    vector_dict: dict[tuple[int, int], tuple[int, int]]

    def __init__(
        self,
        walls: arcade.SpriteList,
        width: int,
        height: int,
    ) -> None: ...
    def pixel_to_tile_pos(self, position: tuple[float, float]) -> tuple[int, int]: ...
    def recalculate_map(
        self, player_pos: tuple[float, float], player_view_distance: int
    ) -> list[tuple[int, int]]: ...
