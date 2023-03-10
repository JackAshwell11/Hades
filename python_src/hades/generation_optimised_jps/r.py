import random
import time
from itertools import pairwise

import numpy as np
from hades.constants.generation import TileType
from hades.generation_optimised_jps.astar_jps_recursive import (
    calculate_astar_path as jps_rec,
)
from hades.generation_optimised_jps.map import create_map
from hades.generation_optimised_jps.old_astar import calculate_astar_path as old
from hades.generation_optimised_jps.primitives import Point


def run_jps():
    grid = [
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
        [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
        [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
    ]
    n = 10
    seed = random.randint(0, 1000000000000000)
    print(f"Seed={seed}")
    rand_gen = random.Random(seed)
    for _ in range(n):
        grid[rand_gen.randint(0, len(grid) - 1)][
            rand_gen.randint(0, len(grid[0]) - 1)
        ] = TileType.OBSTACLE
    start = Point(
        rand_gen.randint(0, len(grid[0]) - 1), rand_gen.randint(0, len(grid) - 1)
    )
    end = Point(
        rand_gen.randint(0, len(grid[0]) - 1), rand_gen.randint(0, len(grid) - 1)
    )
    grid = np.array(grid)
    grid[start.y][start.x] = -2
    grid[end.y][end.x] = -3
    print(f"start grid=\n{grid}")
    print(f"start={start} ({grid[start.y][start.x]}), end={end} ({grid[end.y][end.x]})")
    print("************START CALCULATE**************")
    res = jps_rec(grid, start, end)
    for i in res:
        grid[i.y][i.x] = -5
    grid[start.y][start.x] = -2
    grid[end.y][end.x] = -3
    print("************END CALCULATE**************")
    print(f"result path={list(reversed(res))}")
    print(f"result grid=\n{grid}")


def benchmark():
    n = 10
    result_old = []
    result_jps_rec = []

    for i in range(n):
        print(f"Doing iteration {i+1} of {n}")
        rooms, grid = create_map(500)

        start_old = time.perf_counter()
        for conn in pairwise(rooms):
            old(grid, conn[0].center, conn[1].center)
        result_old.append(time.perf_counter() - start_old)

        start_jps_rec = time.perf_counter()
        for conn in pairwise(rooms):
            jps_rec(grid, conn[0].center, conn[1].center)
        result_jps_rec.append(time.perf_counter() - start_jps_rec)

        print(
            f"Jump point search took {sum(result_jps_rec)/n} and A* took"
            f" {sum(result_old)/n}"
        )


run_jps()
# benchmark()
