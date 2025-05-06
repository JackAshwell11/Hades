add_executable(${TEST_MODULE}
        ${CMAKE_SOURCE_DIR}/tests/ecs/test_registry.cpp
        ${CMAKE_SOURCE_DIR}/tests/ecs/test_steering.cpp
        ${CMAKE_SOURCE_DIR}/tests/ecs/systems/test_armour_regen.cpp
        ${CMAKE_SOURCE_DIR}/tests/ecs/systems/test_attacks.cpp
        ${CMAKE_SOURCE_DIR}/tests/ecs/systems/test_effects.cpp
        ${CMAKE_SOURCE_DIR}/tests/ecs/systems/test_inventory.cpp
        ${CMAKE_SOURCE_DIR}/tests/ecs/systems/test_movements.cpp
        ${CMAKE_SOURCE_DIR}/tests/ecs/systems/test_physics.cpp
        ${CMAKE_SOURCE_DIR}/tests/ecs/systems/test_upgrade.cpp
        ${CMAKE_SOURCE_DIR}/tests/generation/test_bsp.cpp
        ${CMAKE_SOURCE_DIR}/tests/generation/test_dijkstra.cpp
        ${CMAKE_SOURCE_DIR}/tests/generation/test_map.cpp
        ${CMAKE_SOURCE_DIR}/tests/generation/test_primitives.cpp
        ${CMAKE_SOURCE_DIR}/tests/test_factories.cpp
        ${CMAKE_SOURCE_DIR}/tests/test_game_engine.cpp
)
target_include_directories(${TEST_MODULE} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(${TEST_MODULE}
        PRIVATE ${CPP_LIB}
        PRIVATE GTest::gtest_main
)
add_test(NAME Tests COMMAND ${TEST_MODULE})
