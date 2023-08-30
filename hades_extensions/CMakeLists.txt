# Define the minimum CMake version
cmake_minimum_required(VERSION 3.18.2)

# Define some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Get a path to the include directory as well as all the source and test files
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/astar.cpp
        ${CMAKE_SOURCE_DIR}/src/bsp.cpp
        ${CMAKE_SOURCE_DIR}/src/map.cpp
        ${CMAKE_SOURCE_DIR}/src/primitives.cpp
        ${CMAKE_SOURCE_DIR}/src/steering.cpp
)
set(HEADER_FILES
        ${CMAKE_SOURCE_DIR}/include/astar.hpp
        ${CMAKE_SOURCE_DIR}/include/bsp.hpp
        ${CMAKE_SOURCE_DIR}/include/map.hpp
        ${CMAKE_SOURCE_DIR}/include/primitives.hpp
        ${CMAKE_SOURCE_DIR}/include/steering.hpp
)
set(TEST_FILES
        ${CMAKE_SOURCE_DIR}/test/test_astar.cpp
        ${CMAKE_SOURCE_DIR}/test/test_bsp.cpp
        ${CMAKE_SOURCE_DIR}/test/test_map.cpp
        ${CMAKE_SOURCE_DIR}/test/test_primitives.cpp
        ${CMAKE_SOURCE_DIR}/test/test_steering.cpp
)

# Define the module names and initialise the project
set(PY_MODULE hades_extensions)
set(CPP_LIB hades_extensions_lib)
set(TEST_MODULE hades_extensions_tests)
project(${PY_MODULE} LANGUAGES CXX)

# Add the src and test directories to the project
include(FetchContent)
add_subdirectory(src)
add_subdirectory(test)
