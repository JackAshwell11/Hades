# Fetch the GoogleTest repository and initialise it
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
FetchContent_MakeAvailable(googletest)

# Create an executable from our test code and link it against GTest and the C++ library
include(CTest)
add_executable(${TEST_MODULE}
        ${CMAKE_SOURCE_DIR}/tests/game_objects/test_registry.cpp
        ${CMAKE_SOURCE_DIR}/tests/game_objects/test_steering.cpp
        ${CMAKE_SOURCE_DIR}/tests/game_objects/systems/test_armour_regen.cpp
        ${CMAKE_SOURCE_DIR}/tests/game_objects/systems/test_attacks.cpp
        ${CMAKE_SOURCE_DIR}/tests/game_objects/systems/test_effects.cpp
        ${CMAKE_SOURCE_DIR}/tests/game_objects/systems/test_inventory.cpp
        ${CMAKE_SOURCE_DIR}/tests/game_objects/systems/test_movements.cpp
        ${CMAKE_SOURCE_DIR}/tests/game_objects/systems/test_upgrade.cpp
        ${CMAKE_SOURCE_DIR}/tests/generation/test_astar.cpp
        ${CMAKE_SOURCE_DIR}/tests/generation/test_bsp.cpp
        ${CMAKE_SOURCE_DIR}/tests/generation/test_map.cpp
        ${CMAKE_SOURCE_DIR}/tests/generation/test_primitives.cpp
)
target_include_directories(${TEST_MODULE} PUBLIC ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(
        ${TEST_MODULE}
        gtest_main
        ${CPP_LIB}
)

# Add the executable as a test to CTest
add_test(NAME Tests COMMAND ${TEST_MODULE})
