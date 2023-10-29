// Ensure this file is only included once
#pragma once

// Std headers
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

// Local headers
#include "hash_combine.hpp"

// ----- ENUMS ------------------------------
/// Stores the different types of tiles in the game map.
enum class TileType {
  Empty,
  Floor,
  Wall,
  Obstacle,
  Player,
  Potion,
};

// ----- STRUCTURES ------------------------------
/// Represents a 2D position.
struct Position {
  inline bool operator==(const Position &position) const { return x == position.x && y == position.y; }

  inline bool operator!=(const Position &position) const { return x != position.x || y != position.y; }

  inline Position operator+(const Position &position) const { return {x + position.x, y + position.y}; }

  inline Position operator-(const Position &position) const {
    return {std::abs(x - position.x), std::abs(y - position.y)};
  }

  /// The x position of the position.
  int x;

  /// The y position of the position.
  int y;

  /// The default constructor.
  Position() = default;

  /// Initialise the object.
  ///
  /// @param x - The x position of the position.
  /// @param y - The y position of the position.
  Position(int x, int y) : x(x), y(y) {}
};

/// Represents a 2D grid with a set width and height through a 1D vector.
struct Grid {
  /// The width of the 2D grid.
  int width;

  /// The height of the 2D grid.
  int height;

  /// The vector which represents the 2D grid.
  std::unique_ptr<std::vector<TileType>> grid;

  /// Initialise the object.
  ///
  /// @param width_val - The width of the 2D grid.
  /// @param height_val - The height of the 2D grid.
  Grid(int width_val, int height_val)
      : width(width_val),
        height(height_val),
        grid(std::make_unique<std::vector<TileType>>(width * height, TileType::Empty)) {}

  /// Convert a 2D grid position to a 1D grid position.
  ///
  /// @param pos - The position to convert.
  /// @throws std::out_of_range - Position must be within range.
  /// @return The 1D grid position.
  [[nodiscard]] inline int convert_position(const Position &pos) const {
    if (pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height) {
      throw std::out_of_range("Position must be within range");
    }
    return width * pos.y + pos.x;
  }

  /// Get a value in the 2D grid from a given position.
  ///
  /// @param pos - The position to get the value for.
  /// @throws std::out_of_range - Position must be within range.
  /// @return The value at the given position.
  [[nodiscard]] inline TileType get_value(const Position &pos) const { return grid->at(convert_position(pos)); }

  /// Set a value in the 2D grid from a given position.
  ///
  /// @param pos - The position to set.
  /// @throws std::out_of_range - Position must be within range.
  inline void set_value(const Position &pos, const TileType target) const { grid->at(convert_position(pos)) = target; }
};

/// Represents a rectangle in 2D space.
struct Rect {
  inline bool operator==(const Rect &rect) const {
    return top_left == rect.top_left && bottom_right == rect.bottom_right;
  }

  inline bool operator!=(const Rect &rect) const {
    return top_left != rect.top_left || bottom_right != rect.bottom_right;
  }

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
  Rect(const Position &top_left, const Position &bottom_right)
      : top_left(top_left),
        bottom_right(bottom_right),
        centre(static_cast<int>(std::round((top_left + bottom_right).x / 2.0)),
               static_cast<int>(std::round((top_left + bottom_right).y / 2.0))),
        width((top_left - bottom_right).x),
        height((top_left - bottom_right).y) {}

  /// Get the Chebyshev distance to another rect.
  ///
  /// @param other - The rect to find the distance to.
  /// @return The Chebyshev distance between this rect and the given rect.
  [[nodiscard]] inline int get_distance_to(const Rect &other) const {
    return std::max(abs(centre.x - other.centre.x), abs(centre.y - other.centre.y));
  }

  /// Place the rect in the 2D grid.
  ///
  /// @details It is the responsibility of the caller to ensure that the rect fits in the grid.
  /// @param grid - The 2D grid which represents the dungeon.
  void place_rect(Grid &grid) const;
};

// ----- HASHES ------------------------------
template <>
struct std::hash<Position> {
  std::size_t operator()(const Position &position) const {
    std::size_t res = 0;
    hash_combine(res, position.x);
    hash_combine(res, position.y);
    return res;
  }
};

template <>
struct std::hash<Rect> {
  std::size_t operator()(const Rect &rect) const {
    std::size_t res = 0;
    hash_combine(res, rect.top_left);
    hash_combine(res, rect.bottom_right);
    return res;
  }
};
