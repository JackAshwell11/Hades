import time

import arcade

import hades.extensions.vector_field.vector_field as c_field
import hades.vector_field as py_field
from hades.constants.generation import TileType
from hades.generation.map import create_map
from hades.textures import grid_pos_to_pixel

N = 100
c_time = []
py_time = []

for i in range(N):
    print(f"Doing iteration {i+1} of {N}")
    print("before map")
    game_map, constants = create_map(500)

    spritelist = arcade.SpriteList()
    player = (-1, -1)

    print("before loop")
    for count_y, y in enumerate(game_map.grid):
        for count_x, x in enumerate(y):
            if x == TileType.WALL:
                pos_x, pos_y = grid_pos_to_pixel(count_x, count_y)
                spritelist.append(arcade.Sprite(center_x=pos_x, center_y=pos_y))
            elif x == TileType.PLAYER:
                player = grid_pos_to_pixel(count_x, count_y)

    print("before c")
    start = time.perf_counter()
    c = c_field.VectorField(spritelist, constants.width, constants.height)
    c.recalculate_map(player, 5)
    c_time.append(time.perf_counter() - start)
    print("after c")

    start = time.perf_counter()
    py = py_field.VectorField(constants.width, constants.height, spritelist)
    py.recalculate_map(player, 5)
    py_time.append(time.perf_counter() - start)
    print("after py")

print(
    f"C++ implementation too {sum(c_time)/N} seconds whereas Python implementation took"
    f" {sum(py_time)/N}. Which is {(sum(c_time)/N)/(sum(py_time)/N)}x faster"
)
