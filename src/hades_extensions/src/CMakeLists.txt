# Create the C++ library which will be used for the tests and the Python module and link
# it against Chipmunk
add_library(${CPP_LIB} STATIC
        ${CMAKE_SOURCE_DIR}/src/ecs/registry.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/steering.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/armour_regen.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/attacks.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/effects.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/inventory.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/movements.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/physics.cpp
        ${CMAKE_SOURCE_DIR}/src/ecs/systems/upgrade.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/bsp.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/dijkstra.cpp
        ${CMAKE_SOURCE_DIR}/src/generation/map.cpp
)
target_include_directories(${CPP_LIB}
        PUBLIC ${CMAKE_SOURCE_DIR}/include
        PRIVATE ${chipmunk_SOURCE_DIR}/include
)
target_link_libraries(${CPP_LIB}
        PRIVATE unofficial::chipmunk::chipmunk
        PRIVATE pybind11::module
)

# Create the Python module
if (DO_PYTHON)
    message(STATUS "Python bindings enabled (DO_PYTHON is set)")
    pybind11_add_module(${PY_MODULE} ${CMAKE_SOURCE_DIR}/src/binding.cpp)
    target_link_libraries(${PY_MODULE} PRIVATE ${CPP_LIB})
    install(TARGETS ${PY_MODULE} LIBRARY DESTINATION .)
    get_target_property(CHIPMUNK_DLL unofficial::chipmunk::chipmunk LOCATION)
    install(FILES ${CHIPMUNK_DLL} DESTINATION .)
else ()
    message(STATUS "Python bindings disabled (DO_PYTHON is not set)")
endif ()
