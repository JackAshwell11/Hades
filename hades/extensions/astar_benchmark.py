import jps_py_astar as jpy
import old_py_astar as opy


def benchmark():
    import time

    from hades.generation.map import Map

    N = 100
    resulto = []
    resultj = []

    for i in range(N):
        print(f"Doing iteration {i+1} of {N}")
        map_obj = Map(5)
        rooms = map_obj.split_bsp().generate_rooms()
        grid = map_obj.create_hallways(rooms)

        start = time.perf_counter()
        opy.calculate_astar_path(grid, rooms[0].center, rooms[-1].center)
        resulto.append(time.perf_counter() - start)

        start = time.perf_counter()
        jpy.calculate_astar_path(grid, rooms[0].center, rooms[-1].center)
        resultj.append(time.perf_counter() - start)
        print(f"Jump point search took {resultj[-1]} and A* took {resulto[-1]}")

    print(
        f"Jump point search took {sum(resultj)/N} and A* took {sum(resulto)/N}. Which"
        f" is {(sum(resulto)/N)/sum(resultj)/N}x faster"
    )


def run_jps():
    import numpy as np
    from hades.generation.primitives import Point
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
    N = 10
    import random
    for _ in range(N):
        grid[random.randint(0, len(grid) - 1)][random.randint(0, len(grid[0]) - 1)] = -1
    start = Point(random.randint(0, len(grid[0])-1), random.randint(0, len(grid)-1))
    end = Point(random.randint(0, len(grid[0])-1), random.randint(0, len(grid)-1))
    grid = np.array(grid)
    res = jpy.calculate_astar_path(grid, start, end)
    for i in res:
        grid[i.y][i.x] = -5
    grid[start.y][start.x] = -2
    grid[end.y][end.x] = -3
    if not res:
        g = []
        for u in grid:
            g.append(list(u))
        print(g)
    else:
        print(grid)


#run_jps()
benchmark()
