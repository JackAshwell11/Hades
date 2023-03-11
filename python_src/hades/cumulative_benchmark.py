import time

from hades.generation.map import create_map as org
from hades.generation_optimised.map import create_map as optimised
from hades_extensions import create_map as cpp
from hades_extensions_rust import create_map as rust

n = 1000

arr_org = []
arr_optimised = []
arr_rust = []
arr_cpp = []

for i in range(n):
    print(f"Iteration {i+1} of {n}")

    start_org = time.perf_counter()
    org(500, i)
    arr_org.append(time.perf_counter() - start_org)

    start_optimised = time.perf_counter()
    optimised(500, i)
    arr_optimised.append(time.perf_counter() - start_optimised)

    start_rust = time.perf_counter()
    rust(500, i)
    arr_rust.append(time.perf_counter() - start_rust)

    start_cpp = time.perf_counter()
    cpp(500, i)
    arr_cpp.append(time.perf_counter() - start_cpp)

    print(
        f"""Org took {sum(arr_org)/len(arr_org)},
Optimised took {sum(arr_optimised)/len(arr_optimised)},
Rust took {sum(arr_rust)/len(arr_rust)},
Cpp took {sum(arr_cpp)/len(arr_cpp)}\n"""
    )
