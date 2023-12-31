# Fetch the pybind11 repository and initialise it
find_package(Python COMPONENTS Interpreter Development REQUIRED)
FetchContent_Declare(
        pybind11_source
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
)
FetchContent_GetProperties(pybind11_source)
FetchContent_Populate(pybind11_source)
add_subdirectory(
        ${pybind11_source_SOURCE_DIR}
        ${pybind11_source_BINARY_DIR}
)

# Create the C++ library which will be used for the tests and the Python module
add_library(${CPP_LIB} STATIC
        ${CMAKE_SOURCE_DIR}/src/game_objects/registry.cpp
        ${CMAKE_SOURCE_DIR}/src/game_objects/steering.cpp
        ${CMAKE_SOURCE_DIR}/src/game_objects/systems/armour_regen.cpp
        ${CMAKE_SOURCE_DIR}/src/game_objects/systems/attacks.cpp
        ${CMAKE_SOURCE_DIR}/src/game_objects/systems/effects.cpp
        ${CMAKE_SOURCE_DIR}/src/game_objects/systems/inventory.cpp
        ${CMAKE_SOURCE_DIR}/src/game_objects/systems/movements.cpp
        ${CMAKE_SOURCE_DIR}/src/game_objects/systems/upgrade.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/bsp.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/map.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/searching.cpp
)
target_include_directories(${CPP_LIB} PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Create the Python module
if (DO_PYTHON)
    pybind11_add_module(${PY_MODULE} ${CMAKE_SOURCE_DIR}/src/binding.cpp)
    target_link_libraries(${PY_MODULE} PRIVATE ${CPP_LIB})
else ()
    message(STATUS "Python bindings disabled (DO_PYTHON is not set)")
endif ()
