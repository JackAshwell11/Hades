// Ensure this file is only included once
#pragma once

// Std includes
#include <random>

// Custom includes
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
/// A binary spaced partition leaf used to generate the dungeon's rooms.
///
/// Attributes
/// ----------
/// left - The left container of this leaf. If this is null, we have reached the
/// end of the branch.
/// right - The right container of this leaf. If this is null, we have reached
/// the end of the branch.
/// room - The rect object for representing the room inside this leaf.
struct Leaf {
  // Parameters
  Rect container{};

  // Attributes
  Leaf *left{}, *right{};
  Rect *room{};

  inline bool operator==(const Leaf lef) const {
    return container == lef.container && left == lef.left && right == lef.right;
  }

  /// Default constructor for a Leaf object. This should not be used.
  Leaf() = default;

  /// Constructs a Leaf object.
  ///
  /// Parameters
  /// ----------
  /// container - The rect object for representing this leaf.
  ///
  /// Returns
  /// -------
  /// A Leaf object.
  explicit Leaf(Rect container_val) { container = container_val; }

  /// Split a container either horizontally or vertically.
  ///
  /// Parameters
  /// ----------
  /// grid - The 2D grid which represents the dungeon.
  /// random_generator - The random generator used to generate the bsp.
  /// debug_game - Whether the game is in debug mode or not.
  ///
  /// Returns
  /// -------
  /// Whether the split was successful or not.
  bool split(Grid &grid, std::mt19937 &random_generator, bool debug_game);

  /// Create a random sized room inside a container.
  ///
  /// Parameters
  /// ----------
  /// grid - The 2D grid which represents the dungeon.
  /// random_generator - The random generator used to generate the bsp.
  ///
  /// Returns
  /// -------
  /// Whether the room creation was successful or not.
  bool create_room(Grid &grid, std::mt19937 &random_generator);
};
