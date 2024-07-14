// Ensure this file is only included once
#pragma once

// External headers
#ifdef Py_DEBUG
#undef Py_DEBUG
#include <pybind11/pytypes.h>
#define Py_DEBUG
#else
#include <pybind11/pytypes.h>
#endif

// Local headers
#include "game_objects/types.hpp"

// ----- COMPONENTS ------------------------------
/// Allows a game object to hold a reference to the Python sprite object.
struct PythonSprite final : ComponentBase {
  /// The reference to the Python sprite object.
  pybind11::handle sprite;
};
