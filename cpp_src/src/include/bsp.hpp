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
/// split_vertical - Whether the leaf was split vertically or not.
struct Leaf {
  // Parameters
  Rect container{};

  // Attributes
  Leaf *left{}, *right{};
  Rect *room{};
  bool split_vertical{};

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
  /// min_container_size - The minimum size one side of a container can be.
  /// debug_game - Whether the game is in debug mode or not.
  ///
  /// Returns
  /// -------
  /// Whether the split was successful or not.
  bool split(std::vector<std::vector<TileType>> &grid, std::mt19937 &random_generator, bool debug_game);

  bool create_room(std::vector<std::vector<TileType>> &grid, std::mt19937 &random_generator);
};
