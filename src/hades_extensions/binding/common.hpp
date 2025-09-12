// Ensure this file is only included once
#pragma once

// Make pybind11 display detailed error messages
#define PYBIND11_DETAILED_ERROR_MESSAGES

// External headers
#ifdef Py_DEBUG
#undef Py_DEBUG
#include <pybind11/stl.h>
#define Py_DEBUG
#else
#include <pybind11/stl.h>
#endif

// The declarations for the binding functions
void bind_ecs(const pybind11::module& module);
void bind_systems(const pybind11::module_& module);
