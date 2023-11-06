# Define the minimum CMake version
cmake_minimum_required(VERSION 3.18.2)

# Define some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define the module names and initialise the project
set(PY_MODULE hades_extensions)
set(CPP_LIB hades_extensions_lib)
set(TEST_MODULE hades_extensions_tests)
project(${PY_MODULE} LANGUAGES CXX)

# Enable coverage if supported by the compiler
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Compiler is GNU or Clang, coverage enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
else ()
    message(STATUS "Compiler is not GNU or Clang, coverage disabled")
endif ()

# TODO: Clang-tidy just fails the build for some reason
## Enable clang-tidy if found
#find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
#if (CLANG_TIDY_EXE)
#    message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}")
#    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
#else ()
#    message(STATUS "No clang-tidy found")
#endif ()

## Enable cppcheck if found
#find_program(CPPCHECK_EXE NAMES "cppcheck")
#if (CPPCHECK_EXE)
#    message(STATUS "Found cppcheck: ${CPPCHECK_EXE}")
#    set(CMAKE_CXX_CPPCHECK "${CPPCHECK_EXE}")
#    list(APPEND CMAKE_CXX_CPPCHECK "--enable=all" "--suppress=missingInclude" "--suppress=missingIncludeSystem")
#else ()
#    message(STATUS "No cppcheck found")
#endif ()

# Add the subdirectories for the different parts of the project
include(FetchContent)
add_subdirectory(src)
add_subdirectory(test)

# TODO: Get coverage fully working (could get temporary github repo to test it)
# TODO: Try catch2
# TODO: Try out vcpkg (idk)

# TODO: Should clang-tidy run in CI and local or just CI or just local? If so, should a new action be made or integrated into an existing one?
# TODO: Add stub files and modify Python game to use new ECS
