# Define the minimum CMake version
cmake_minimum_required(VERSION 3.18.2)

# Define some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Get a path to the include directory as well as all the source and test files
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(BINDING_PATH ${CMAKE_SOURCE_DIR}/src/binding.cpp)
set(HEADER_FILES
        ${INCLUDE_PATH}/base.hpp
        ${INCLUDE_PATH}/game_objects/components.hpp
        ${INCLUDE_PATH}/game_objects/registry.hpp
        ${INCLUDE_PATH}/game_objects/steering.hpp
        ${INCLUDE_PATH}/generation/astar.hpp
        ${INCLUDE_PATH}/generation/bsp.hpp
        ${INCLUDE_PATH}/generation/map.hpp
        ${INCLUDE_PATH}/generation/primitives.hpp
)
set(SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/game_objects/components.cpp
        ${CMAKE_SOURCE_DIR}/src/game_objects/registry.cpp
        ${CMAKE_SOURCE_DIR}/src/game_objects/steering.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/astar.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/bsp.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/map.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/primitives.cpp
)
set(TEST_FILES
        ${CMAKE_SOURCE_DIR}/test/game_objects/test_components.cpp
        ${CMAKE_SOURCE_DIR}/test/game_objects/test_registry.cpp
        ${CMAKE_SOURCE_DIR}/test/game_objects/test_steering.cpp
        ${CMAKE_SOURCE_DIR}/test/generation/test_astar.cpp
        ${CMAKE_SOURCE_DIR}/test/generation/test_bsp.cpp
        ${CMAKE_SOURCE_DIR}/test/generation/test_map.cpp
        ${CMAKE_SOURCE_DIR}/test/generation/test_primitives.cpp
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
