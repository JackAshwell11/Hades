# Define the minimum CMake version
cmake_minimum_required(VERSION 3.18.2)

# Define some module names which are used by the C++ library, the tests, and the
# Python module
set(PY_MODULE hades_extensions)
set(CPP_LIB hades_extensions_lib)
set(TEST_MODULE hades_extensions_tests)

# Define a boolean which determines if the tests should be compiled with the C++
# library
SET(DO_TESTS true CACHE BOOL "Compiles the tests with the C++ library")

# Define the project
project(${PY_MODULE} LANGUAGES CXX)

# Initialise some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the src and test directories to the project
add_subdirectory(src)
if (DO_TESTS)
    add_subdirectory(test)
endif()

# TODO: https://github.com/sdispater/pendulum/blob/master/meson.build
# TODO: https://mesonbuild.com/CMake-module.html

