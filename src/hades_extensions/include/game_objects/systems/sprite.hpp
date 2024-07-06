// Ensure this file is only included once
#pragma once

// External headers
#include <pybind11/pytypes.h>

// Local headers
#include "game_objects/types.hpp"

// ----- COMPONENTS ------------------------------
/// Allows a game object to hold a reference to the Python sprite object.
struct PythonSprite final : ComponentBase {
  /// The reference to the Python sprite object.
  pybind11::handle sprite;

  /// Initialise the object.
  explicit PythonSprite() : sprite(pybind11::none()) {}

  /// Set the sprite object.
  ///
  /// @param py_sprite The sprite object to set.
  void set_sprite(const pybind11::handle py_sprite) { sprite = py_sprite; }
};
