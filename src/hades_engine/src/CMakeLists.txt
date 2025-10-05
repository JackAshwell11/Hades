add_library(${CPP_LIB} STATIC
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/armour_regen.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/attacks.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/effects.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/inventory.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/movements.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/physics.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/shop.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/registry.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/steering.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/bsp.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/dijkstra.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/map.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/primitives.cpp
        ${CMAKE_SOURCE_DIR}/src/factories.cpp
        ${CMAKE_SOURCE_DIR}/src/game_engine.cpp
        ${CMAKE_SOURCE_DIR}/src/game_state.cpp
        ${CMAKE_SOURCE_DIR}/src/input_handler.cpp
)
target_include_directories(${CPP_LIB}
        PUBLIC ${CMAKE_SOURCE_DIR}/include
        PRIVATE ${chipmunk_SOURCE_DIR}/include
)
target_link_libraries(${CPP_LIB}
        PUBLIC chipmunk2d::chipmunk2d
        PUBLIC nlohmann_json::nlohmann_json
)
