# Define the minimum CMake version
cmake_minimum_required(VERSION 3.25.2)

# Define some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ENABLE_COVERAGE OFF CACHE BOOL "Enable coverage reporting for GCC/Clang")
set(ENABLE_STATIC_ANALYSIS OFF CACHE BOOL "Enable static analysis using clang-tidy and cppcheck")
set(DO_TESTS ON CACHE BOOL "Enable testing")

# Define the module names and initialise the project
set(PY_MODULE hades_extensions)
set(CPP_LIB hades_extensions_lib)
set(TEST_MODULE hades_extensions_tests)
project(${PY_MODULE} LANGUAGES CXX)

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

# Enable static analysis if enabled by the user
if (ENABLE_STATIC_ANALYSIS)
    # Enable clang-tidy if found
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
    if (CLANG_TIDY_EXE)
        message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    else ()
        message(STATUS "No clang-tidy found")
    endif ()

    # Enable cppcheck if found
    find_program(CPPCHECK_EXE NAMES "cppcheck")
    if (CPPCHECK_EXE)
        message(STATUS "Found cppcheck: ${CPPCHECK_EXE}")
        set(CMAKE_CXX_CPPCHECK "${CPPCHECK_EXE}")
        list(APPEND CMAKE_CXX_CPPCHECK "--enable=all" "--suppress=missingInclude" "--suppress=missingIncludeSystem")
    else ()
        message(STATUS "No cppcheck found")
    endif ()
else ()
    message(STATUS "Static analysis disabled (ENABLE_STATIC_ANALYSIS is not set)")
endif ()

# Add the subdirectories for the different parts of the project
include(FetchContent)
add_subdirectory(src)
if (DO_TESTS)
    add_subdirectory(tests)
endif ()

# TODO: fix the ci build issues with clang-tidy
# TODO: add the cmake presets to the build.py python script
# TODO: add c++ coverage support to the ci build
# TODO: rewrite the python templates for game objects
# TODO: change the components to pass by value and use std::move to signify that ownership is being transferred
# TODO: maybe add vcpkg to the c++ project
