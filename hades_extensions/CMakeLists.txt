# Define the minimum CMake version
cmake_minimum_required(VERSION 3.18.2)

# Define an environment variable to determine if we want to make a Python extension or not
SET(DO_PYTHON false CACHE BOOL "Creates a Python extension for the C++ library")

# Define some module names which are used by the C++ library, the tests, and the
# Python module
set(PY_MODULE hades_extensions)
set(CPP_LIB hades_extensions_lib)
set(TEST_MODULE hades_extensions_tests)

# Initialise some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the project
project(${PY_MODULE} LANGUAGES CXX)

# Initialise some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the src and test directories to the project
add_subdirectory(src)
add_subdirectory(test)
