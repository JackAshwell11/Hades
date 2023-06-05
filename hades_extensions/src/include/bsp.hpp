// Ensure this file is only included once
#pragma once

// Std includes
#include <random>

// Custom includes
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
/// A binary spaced partition leaf used to generate the dungeon's rooms.
///
/// @details left - The left container of this leaf. If this is null, we
/// have reached the end of the branch.
/// @details right - The right container of this leaf. If this is null,
/// we have reached the end of the branch.
/// @details room - The rect object for representing the room inside this
/// leaf.
struct Leaf {
  // Parameters
  std::unique_ptr<Rect> container;

  // Attributes
  std::unique_ptr<Leaf> left, right;
  std::unique_ptr<Rect> room;

  inline bool operator==(const Leaf &lef) const {
    return container == lef.container && left == lef.left && right == lef.right;
  }

  explicit Leaf(Rect container_val) : container(std::make_unique<Rect>(container_val)) {}

  /// Split a container either horizontally or vertically.
  ///
  /// @param grid - The 2D grid which represents the dungeon.
  /// @param random_generator - The random generator used to generate the bsp.
  /// @param debug_game - Whether the game is in debug mode or not.
  /// @return Whether the split was successful or not.
  bool split(Grid &grid, std::mt19937 &random_generator, bool debug_game);

  /// Create a random sized room inside a container.
  ///
  /// @param grid - The 2D grid which represents the dungeon.
  /// @param random_generator - The random generator used to generate the bsp.
  /// @return Whether the room creation was successful or not.
  bool create_room(Grid &grid, std::mt19937 &random_generator);
};
