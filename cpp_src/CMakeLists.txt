# Define the minimum CMake version
cmake_minimum_required(VERSION 3.18.2)

# Define the project
project(hades_extensions LANGUAGES CXX)

# Initialise some environment variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create a shared library which is used by the python module and the tests
set(CPP_LIB hades_extensions_lib)
set(PY_MODULE hades_extensions_py)
set(TEST_MODULE hades_extensions_tests)

# Add the src and test directories to the project
add_subdirectory(src)
add_subdirectory(test)

# Enable support for unit testing in the "test" subdirectory
enable_testing()


# TODO: https://github.com/sdispater/pendulum/blob/master/meson.build
# TODO: https://mesonbuild.com/CMake-module.html
