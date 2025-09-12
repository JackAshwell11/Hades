pybind11_add_module(${PY_MODULE}
        ${CMAKE_SOURCE_DIR}/binding/ecs/ecs.cpp
        ${CMAKE_SOURCE_DIR}/binding/ecs/systems.cpp
        ${CMAKE_SOURCE_DIR}/binding/bindings.cpp
)
target_include_directories(${PY_MODULE} PRIVATE ${CMAKE_SOURCE_DIR}/binding)
target_link_libraries(${PY_MODULE} PRIVATE ${CPP_LIB})
if (DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS ${PY_MODULE} LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR})
endif ()
