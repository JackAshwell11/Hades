# Determine if coverage can be used
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("-- GCC detected, coverage will use gcov")
    set(CMAKE_C_FLAGS "-g -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "-g -fprofile-arcs -ftest-coverage")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message("-- MSVC detected, coverage will use llvm-cov")
    set(CMAKE_C_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
endif ()

# Fetch the GoogleTest repository and initialise it
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

# Create an executable from our test code
add_executable(${TEST_MODULE} test_astar.cpp test_bsp.cpp test_map.cpp test_primitives.cpp)

# Link the test executable against GoogleTest and the C++ shared library
target_link_libraries(
        ${TEST_MODULE}
        gtest_main
        ${CPP_LIB}
)
add_test(NAME Tests COMMAND ${TEST_MODULE})
