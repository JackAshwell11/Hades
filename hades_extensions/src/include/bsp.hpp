// Ensure this file is only included once
#pragma once

// Std includes
#include <random>

// Custom includes
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
/// A binary spaced partition leaf used to generate the dungeon's rooms.
///
/// @param container - The rect object that represents this leaf.
/// @details left - The left container of this leaf. If this is null, we have
/// reached the end of the branch.
/// @details right - The right container of this leaf. If this is null, we have
/// reached the end of the branch.
/// @details room - The rect object for representing the room inside this leaf.
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
};

// ----- FUNCTIONS -------------------------------
/// Split a container either horizontally or vertically.
///
/// @param leaf - The leaf to split.
/// @param random_generator - The random generator used to generate the bsp.
/// @return Whether the split was successful or not.
bool split(Leaf &leaf, std::mt19937 &random_generator);

/// Create a random sized room inside a container.
///
/// @param leaf - The leaf to create a room inside of.
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator used to generate the bsp.
/// @return Whether the room creation was successful or not.
bool create_room(Leaf &leaf, Grid &grid, std::mt19937 &random_generator);
