import time

import matplotlib.pyplot as plt

from game.generation.map import Map

LEVEL_N = 100
ITERATION_N = 50


def loop():
    total_level = []
    total_time = []
    for level in range(LEVEL_N):
        print(f"Doing level {level+1}")
        time_store = []
        total_level.append(level + 1)
        for _ in range(ITERATION_N):
            start = time.perf_counter()
            Map(level + 1)
            time_store.append(time.perf_counter() - start)
        total_time.append(sum(time_store) / ITERATION_N)
        print(f"Finish level {level+1}. Average time is {sum(time_store)/ITERATION_N}")
    print(f"Average time to complete is {sum(total_time)/ITERATION_N}")
    plt.plot(total_level, total_time)
    plt.xlabel("Level")
    plt.ylabel("Time")
    plt.show()


def single_test():
    f = Map(200)
    print(f.grid)


loop()
# single_test()
