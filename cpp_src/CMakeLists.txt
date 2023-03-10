# Define the minimum CMake version
cmake_minimum_required(VERSION 3.18.2)

# Define the project
project(hades_extensions LANGUAGES CXX)

# Initialise some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Python interpreter
find_package(PythonInterp REQUIRED)

# Fetch the pybind11 repository and initialise it
include(FetchContent)
FetchContent_Declare(
        pybind11_source
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.3
)
FetchContent_GetProperties(pybind11_source)
FetchContent_Populate(pybind11_source)
add_subdirectory(
        ${pybind11_source_SOURCE_DIR}
        ${pybind11_source_BINARY_DIR}
)

# Initialise the module with pybind11
pybind11_add_module(hades_extensions src/map.cpp)


# TODO: https://github.com/sdispater/pendulum/blob/master/meson.build
# TODO: https://mesonbuild.com/CMake-module.html
