import time

from cpp_a.hades_extensions import create_map as cpp
from cpp_f.map import create_map as cpp_par
from hades.generation.map import create_map as org
from hades.generation_optimised.map import create_map as optimised
from hades_extensions import create_map as rust
from hades_extensions_par import create_map as rust_par

n = 500

arr_org = []
arr_optimised = []
arr_rust = []
arr_cpp_par = []
arr_cpp = []
arr_rust_par = []

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

    start_cpp_par = time.perf_counter()
    cpp_par(500, i)
    arr_cpp_par.append(time.perf_counter() - start_cpp_par)

    start_cpp = time.perf_counter()
    cpp(500, i)
    arr_cpp.append(time.perf_counter() - start_cpp)

    start_rust_par = time.perf_counter()
    rust_par(500, i)
    arr_rust_par.append(time.perf_counter() - start_rust_par)

    print(
        f"""Org took {sum(arr_org)/n},
Optimised took {sum(arr_optimised)/n},
Rust took {sum(arr_rust)/n},
Rust par took {sum(arr_rust_par)/n},
Cpp took {sum(arr_cpp)/n},
Cpp par took {sum(arr_cpp_par)/n}\n"""
    )
