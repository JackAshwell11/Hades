# Fetch the pybind11 repository and initialise it
find_package(PythonInterp REQUIRED)
FetchContent_Declare(
        pybind11_source
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.3
)
FetchContent_GetProperties(pybind11_source)
FetchContent_Populate(pybind11_source)
add_subdirectory(
        ${pybind11_source_SOURCE_DIR}
        ${pybind11_source_BINARY_DIR}
)

# Create the C++ library which will be used for the tests and pybind11
add_library(${CPP_LIB} STATIC ${SOURCE_FILES})
target_include_directories(${CPP_LIB} PUBLIC ${INCLUDE_DIR})

# Create the Python module
pybind11_add_module(${PY_MODULE} ${CMAKE_SOURCE_DIR}/src/binding.cpp)
target_link_libraries(${PY_MODULE} PRIVATE ${CPP_LIB})
