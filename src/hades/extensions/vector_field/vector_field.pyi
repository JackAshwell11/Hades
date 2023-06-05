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
    @staticmethod
    def pixel_to_grid_pos(
        position: tuple[float, float] | list[float],
    ) -> tuple[int, int]: ...
    def recalculate_map(
        self,
        player_pos: tuple[float, float] | list[float],
        player_view_distance: int,
    ) -> list[tuple[int, int]]: ...
