# Define some project-wide options
cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ENABLE_COVERAGE OFF CACHE BOOL "Enable coverage reporting for GCC/Clang")
set(ENABLE_STATIC_ANALYSIS OFF CACHE BOOL "Enable static analysis using clang-tidy")
set(DO_TESTS ON CACHE BOOL "Enable testing")

# Define the module names and initialise the project
set(PY_MODULE hades_extensions)
set(CPP_LIB hades_extensions_lib)
set(TEST_MODULE hades_extensions_tests)
project(${PY_MODULE} LANGUAGES CXX)

# Check if the tests have been disabled
if (NOT DO_TESTS)
    message(STATUS "Tests disabled (DO_TESTS is not set)")
endif ()

# Enable coverage if supported by the compiler and enabled by the user
if (ENABLE_COVERAGE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "Coverage enabled (compiler is GCC/Clang)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    else ()
        message(STATUS "Coverage disabled (compiler is not GCC/Clang)")
    endif ()
else ()
    message(STATUS "Coverage disabled (ENABLE_COVERAGE is not set)")
endif ()

# Add the subdirectories for the different parts of the project
include(FetchContent)
add_subdirectory(src)
if (DO_TESTS)
    add_subdirectory(tests)
endif ()

# TODO: maybe add vcpkg to the c++ project
# TODO: maybe switch to catch2
# TODO: maybe change the components to pass by value and use std::move to signify that ownership is being transferred (or find another way)
