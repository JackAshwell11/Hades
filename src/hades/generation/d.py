import time

from hades.generation.map import create_map

n = 200

map_time = []

for i in range(n):
    print(f"Doing iteration {i+1} of {n}")
    start = time.perf_counter()
    create_map(1000)
    map_time.append(time.perf_counter() - start)

print(f"It took {sum(map_time)/n} seconds")
