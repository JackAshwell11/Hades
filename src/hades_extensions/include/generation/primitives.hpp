// Ensure this file is only included once
#pragma once

// Std headers
#include <cmath>
#include <vector>

// Local headers
#include "game_object.hpp"

/// Represents a 2D position.
struct Position {
  /// The equality operator.
  auto operator==(const Position& position) const -> bool { return x == position.x && y == position.y; }

  /// The addition operator.
  auto operator+(const Position& position) const -> Position { return {.x = x + position.x, .y = y + position.y}; }

  /// The subtraction operator.
  auto operator-(const Position& position) const -> Position {
    return {.x = std::abs(x - position.x), .y = std::abs(y - position.y)};
  }

  /// The x position of the position.
  int x;

  /// The y position of the position.
  int y;

  /// Get the Chebyshev distance to another position.
  ///
  /// @param other - The other position to find the distance to.
  /// @return The Chebyshev distance between this position and the other position.
  [[nodiscard]] auto get_distance_to(const Position& other) const -> int;
};

/// Represents a rectangle in 2D space.
struct Rect {
  /// The == operator.
  auto operator==(const Rect& rect) const -> bool {
    return top_left == rect.top_left && bottom_right == rect.bottom_right;
  }

  /// The != operator.
  auto operator!=(const Rect& rect) const -> bool { return !(*this == rect); }

  /// The top left position of the rect.
  Position top_left;

  /// The bottom right position of the rect.
  Position bottom_right;

  /// The centre position of the rect.
  Position centre;

  /// The width of the rect.
  int width;

  /// The height of the rect.
  int height;

  /// Initialise the object.
  ///
  /// @param top_left - The top left position of the rect.
  /// @param bottom_right - The bottom right position of the rect.
  Rect(const Position& top_left, const Position& bottom_right);
};

/// Represents a 2D grid with a set width and height through a 1D vector.
struct Grid {
  /// The width of the 2D grid.
  int width;

  /// The height of the 2D grid.
  int height;

  /// The vector which represents the 2D grid.
  std::vector<GameObjectType> grid;

  /// Initialise the object.
  ///
  /// @param width - The width of the 2D grid.
  /// @param height - The height of the 2D grid.
  Grid(int width, int height);

  /// Check if a position is within the 2D grid.
  ///
  /// @param position - The position to check for.
  /// @return Whether the position is within the 2D grid or not.
  [[nodiscard]] auto is_position_within(const Position& position) const -> bool;

  /// Convert a 1D grid position to a 2D grid position.
  ///
  /// @param position - The position to convert.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  /// @return The 2D grid position.
  [[nodiscard]] auto convert_position(int position) const -> Position;

  /// Convert a 2D grid position to a 1D grid position.
  ///
  /// @param position - The position to convert.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  /// @return The 1D grid position.
  [[nodiscard]] auto convert_position(const Position& position) const -> int;

  /// Get a value in the 2D grid from a given position.
  ///
  /// @param position - The position to get the value for.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  /// @return The value at the given position.
  [[nodiscard]] auto get_value(const Position& position) const -> GameObjectType;

  /// Set a value in the 2D grid from a given position.
  ///
  /// @param position - The position to set.
  /// @param target - The value to set at the given position.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  void set_value(const Position& position, GameObjectType target);

  /// Get the neighbours of a given position.
  ///
  /// @param position - The position to get the neighbours for.
  /// @return The neighbours of the given position.
  [[nodiscard]] auto get_neighbours(const Position& position) const -> std::vector<Position>;

  /// Place a rect in the 2D grid.
  ///
  /// @param rect - The rect to place in the 2D grid.
  void place_rect(const Rect& rect);
};

template <>
struct std::hash<Position> {
  auto operator()(const Position& pos) const noexcept -> std::size_t {
    return std::hash<int>{}(pos.x) ^ (std::hash<int>{}(pos.y) << 1);
  }
};
