# Define the minimum CMake version
cmake_minimum_required(VERSION 3.18.2)

# Define some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Get a path to the include directory as well as all the source and test files
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/test)
set(BINDING_PATH ${SRC_DIR}/binding.cpp)
set(HEADER_FILES
        ${INCLUDE_DIR}/base.hpp
        ${INCLUDE_DIR}/macros.hpp
        ${INCLUDE_DIR}/game_objects/components.hpp
        ${INCLUDE_DIR}/game_objects/registry.hpp
        ${INCLUDE_DIR}/game_objects/steering.hpp
        ${INCLUDE_DIR}/game_objects/systems/attacks.hpp
        ${INCLUDE_DIR}/game_objects/systems/attributes.hpp
        ${INCLUDE_DIR}/game_objects/systems/inventory.hpp
        ${INCLUDE_DIR}/game_objects/systems/movements.hpp
        ${INCLUDE_DIR}/generation/astar.hpp
        ${INCLUDE_DIR}/generation/bsp.hpp
        ${INCLUDE_DIR}/generation/map.hpp
        ${INCLUDE_DIR}/generation/primitives.hpp
)
set(SOURCE_FILES
        ${SRC_DIR}/game_objects/registry.cpp
        ${SRC_DIR}/game_objects/steering.cpp
        ${SRC_DIR}/game_objects/systems/attacks.cpp
        ${SRC_DIR}/game_objects/systems/attributes.cpp
        ${SRC_DIR}/game_objects/systems/inventory.cpp
        ${SRC_DIR}/game_objects/systems/movements.cpp
        ${SRC_DIR}/generation/astar.cpp
        ${SRC_DIR}/generation/bsp.cpp
        ${SRC_DIR}/generation/map.cpp
        ${SRC_DIR}/generation/primitives.cpp
)
set(TEST_FILES
        ${TEST_DIR}/game_objects/test_components.cpp
        ${TEST_DIR}/game_objects/test_registry.cpp
        ${TEST_DIR}/game_objects/test_steering.cpp
        ${TEST_DIR}/game_objects/systems/test_attacks.cpp
        ${TEST_DIR}/game_objects/systems/test_attributes.cpp
        ${TEST_DIR}/game_objects/systems/test_inventory.cpp
        ${TEST_DIR}/game_objects/systems/test_movements.cpp
        ${TEST_DIR}/generation/test_astar.cpp
        ${TEST_DIR}/generation/test_bsp.cpp
        ${TEST_DIR}/generation/test_map.cpp
        ${TEST_DIR}/generation/test_primitives.cpp
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

# TODO: Look at maybe adding clang-tidy and clang-format files
