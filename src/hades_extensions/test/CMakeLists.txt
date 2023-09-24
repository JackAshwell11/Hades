# Fetch the GoogleTest repository and initialise it
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.13.0
)
set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
FetchContent_MakeAvailable(googletest)

# Create an executable from our test code linking it against GTest and the C++
# shared library then add it as a test
include(CTest)
add_executable(${TEST_MODULE} ${TEST_FILES})
target_link_libraries(
        ${TEST_MODULE}
        gtest_main
        ${CPP_LIB}
)
add_test(NAME Tests COMMAND ${TEST_MODULE})
