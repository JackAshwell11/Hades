import time

import arcade

import hades.extensions.vector_field.vector_field as c_field
import hades.vector_field as py_field
from hades.constants.generation import TileType
from hades.generation.map import create_map
from hades.textures import grid_pos_to_pixel

N = 100000
c_time = []
py_time = []

for i in range(N):
    print(f"Doing iteration {i+1} of {N}")
    game_map, constants = create_map(1)

    spritelist = arcade.SpriteList()
    player = (-1, -1)

    for count_y, y in enumerate(game_map.grid):
        for count_x, x in enumerate(y):
            if x == TileType.WALL:
                pos_x, pos_y = grid_pos_to_pixel(count_x, count_y)
                spritelist.append(arcade.Sprite(center_x=pos_x, center_y=pos_y))
            elif x == TileType.PLAYER:
                player = grid_pos_to_pixel(count_x, count_y)

    c_start = time.perf_counter()
    c_test = c_field.VectorField(spritelist, constants.width, constants.height)
    c_test.recalculate_map(player, 5)
    c_time.append(time.perf_counter() - c_start)

    py_start = time.perf_counter()
    py_test = py_field.VectorField(constants.width, constants.height, spritelist)
    py_test.recalculate_map(player, 5)
    py_time.append(time.perf_counter() - py_start)
    print(f"Current multiplier: {(sum(py_time) / N) / (sum(c_time) / N)}")

print(
    f"C++ implementation took {sum(c_time) / N} seconds whereas Python implementation"
    f" took {sum(py_time) / N}. Therefore, C++ is"
    f" {(sum(py_time) / N) / (sum(c_time) / N)}x faster"
)
