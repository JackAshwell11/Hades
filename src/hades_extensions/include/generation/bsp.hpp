// Ensure this file is only included once
#pragma once

// Std includes
#include <random>

// Custom includes
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
/// A binary spaced partition leaf used to generate the dungeon's rooms.
struct Leaf {
  inline bool operator==(const Leaf &leaf) const {
    return container == leaf.container && left == leaf.left && right == leaf.right;
  }

  /// The rect object that represents this leaf.
  std::unique_ptr<Rect> container;

  /// The left container of this leaf. If this is null, we have reached the end of the branch.
  std::unique_ptr<Leaf> left;

  /// The right container of this leaf. If this is null, we have reached the end of the branch.
  std::unique_ptr<Leaf> right;

  /// The rect object for representing the room inside this leaf.
  std::unique_ptr<Rect> room;

  /// Initialise the object.
  ///
  /// @param container_val - The rect object that represents this leaf.
  explicit Leaf(const Rect &container_val) : container(std::make_unique<Rect>(container_val)) {}
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

/// TODO: LOOK AT AND MAYBE REMOVE (OR MOVE TO COMMENT IN BSP.HPP)
/// When creating a container, the split wall is included in the rect size,
/// whereas, rooms don't so MIN_CONTAINER_SIZE must be bigger than
/// MIN_ROOM_SIZE.
