# Define some project-wide options
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ENABLE_COVERAGE ON CACHE BOOL "Enable coverage reporting for GCC/Clang")
set(DO_PYTHON ON CACHE BOOL "Enable Python bindings")
set(DO_TESTS ON CACHE BOOL "Enable testing")

# Define the module names and initialise the project
set(PY_MODULE hades_extensions)
set(CPP_LIB hades_extensions_lib)
set(TEST_MODULE hades_extensions_tests)
project(${PY_MODULE} LANGUAGES CXX)

# Set the warning level
if (MSVC)
    add_compile_options(/W4 /WX)
else ()
    add_compile_options(-Wall -Wextra -Werror -pedantic)
endif ()

# Check if the tests have been disabled
if (NOT DO_TESTS)
    message(STATUS "Tests disabled (DO_TESTS is not set)")
endif ()

# Enable coverage if supported by the compiler and enabled by the user
if (ENABLE_COVERAGE)
    if (NOT MSVC)
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
