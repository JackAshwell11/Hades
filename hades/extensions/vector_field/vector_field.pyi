"""Holds stub data for vector_field.cpp to help with type inference."""
from __future__ import annotations

# Pip
import arcade

class VectorField:
    def __init__(
        self,
        sprite_pixel_size: int,
        grid_height: int,
        grid_width: int,
        walls: arcade.SpriteList,
    ) -> None: ...
    def get_tile_pos_for_pixel(
        self, position: tuple[float, float]
    ) -> tuple[int, int]: ...
    def recalculate_map(
        self, player_pos: tuple[float, float], player_view_distance: int
    ) -> list[tuple[int, int]]: ...
    def get_vector_direction(
        self, current_enemy_pos: tuple[float, float]
    ) -> tuple[int, int]: ...
