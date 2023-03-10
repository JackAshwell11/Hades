import time

from hades.generation.map import create_map as org
from hades.generation_optimised.map import create_map as optimised
from hades_extensions import create_map as rust

n = 500

arr_org = []
arr_optimised = []
arr_rust = []

for i in range(n):
    print(f"Iteration {i+1} of {n}")

    start_org = time.perf_counter()
    org(500, i)
    arr_org.append(time.perf_counter() - start_org)

    start_optimised = time.perf_counter()
    optimised(500, i)
    arr_optimised.append(time.perf_counter() - start_optimised)

    start_rust_par_item = time.perf_counter()
    rust(500, i)
    arr_rust.append(time.perf_counter() - start_rust_par_item)

    print(
        f"""Org took {sum(arr_org)/n},
Optimised took {sum(arr_optimised)/n},
Rust par iter took {sum(arr_rust)/n}\n"""
    )
