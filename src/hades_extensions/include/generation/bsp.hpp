// Ensure this file is only included once
#pragma once

// Std headers
#include <random>

// Local headers
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
/// A binary spaced partition leaf used to generate the dungeon's rooms.
struct Leaf {
  /// The rect object that represents this leaf.
  const std::unique_ptr<Rect> container;

  /// The left container of this leaf. If this is null, we have reached the end of the branch.
  std::unique_ptr<Leaf> left{};

  /// The right container of this leaf. If this is null, we have reached the end of the branch.
  std::unique_ptr<Leaf> right{};

  /// The rect object for representing the room inside this leaf.
  std::unique_ptr<Rect> room{};

  /// Initialise the object.
  ///
  /// @param container - The rect object that represents this leaf.
  explicit Leaf(const Rect &container) : container(std::make_unique<Rect>(container)) {}
};

// ----- FUNCTIONS -------------------------------
/// Split a leaf either horizontally or vertically recursively.
///
/// @param leaf - The leaf to split.
/// @param random_generator - The random generator to use.
void split(Leaf &leaf, std::mt19937 &random_generator);

/// Create a random sized room inside a container.
///
/// @param leaf - The leaf to create a room inside of.
/// @param grid - The 2D grid which represents the dungeon.
/// @param random_generator - The random generator to use.
/// @param rooms - The vector of rooms to add the new room to.
void create_room(Leaf &leaf, Grid &grid, std::mt19937 &random_generator, std::vector<Rect> &rooms);
