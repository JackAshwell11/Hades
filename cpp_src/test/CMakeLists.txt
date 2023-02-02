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
add_executable(${TEST_MODULE} main.cpp)

# Link the test executable against GoogleTest and the C++ shared library
target_link_libraries(
        ${TEST_MODULE}
        gtest_main
        ${CPP_LIB}
)
add_test(hadesExtensionsTests ${TEST_MODULE})
