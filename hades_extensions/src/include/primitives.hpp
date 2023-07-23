// Ensure this file is only included once
#pragma once

// Std includes
#include <cmath>
#include <memory>
#include <vector>

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
/// Represents a point in the grid.
///
/// @param x - The x position of the point.
/// @param y - The y position of the point.
struct Point {
  int x, y;

  inline bool operator==(const Point pnt) const {
    return x == pnt.x && y == pnt.y;
  }

  inline bool operator!=(const Point pnt) const {
    return x != pnt.x || y != pnt.y;
  }

  inline Point operator+(const Point pnt) const {
    return {x + pnt.x, y + pnt.y};
  }

  inline Point operator-(const Point pnt) const {
    return {abs(x - pnt.x), abs(y - pnt.y)};
  }

  Point() = default;

  Point(int x_val, int y_val) : x(x_val), y(y_val) {}
};

/// Represents a 2D grid with a set width and height through a 1D vector.
///
/// @param width - The width of the 2D grid.
/// @param height - The height of the 2D grid.
/// @details grid - The vector which represents the 2D grid.
struct Grid {
  // Parameters
  int width{}, height{};

  // Attributes
  std::unique_ptr<std::vector<TileType>> grid;

  Grid() = default;

  Grid(int width_val, int height_val)
      : width(width_val),
        height(height_val),
        grid(std::make_unique<std::vector<TileType>>(width * height, TileType::Empty)) {}

  /// Convert a 2D grid position to a 1D grid position.
  ///
  /// @param pos - The position to convert.
  /// @throws std::out_of_range - Position must be within range.
  /// @return The 1D grid position.
  [[nodiscard]] int convert_position(const Point &pos) const;

  /// Get a value in the 2D grid from a given position.
  ///
  /// @param pos - The position to get the value for.
  /// @throws std::out_of_range - Position must be within range.
  /// @return The value at the given position.
  [[nodiscard]] TileType get_value(const Point &pos) const;

  /// Set a value in the 2D grid from a given position.
  ///
  /// @param pos - The position to set.
  /// @throws std::out_of_range - Position must be within range.
  void set_value(const Point &pos, TileType target) const;
};

/// Represents a rectangle of any size useful for the interacting with the 2D
/// grid.
///
/// When creating a container, the split wall is included in the rect size,
/// whereas, rooms don't so MIN_CONTAINER_SIZE must be bigger than
/// MIN_ROOM_SIZE.
///
/// @param top_left - The top left position of the rect.
/// @param bottom_right - The bottom right position of the rect.
/// @details centre - The centre position of the rect.
/// @details width - The width of the rect.
/// @details height - The height of the rect.
struct Rect {
  // Parameters
  Point top_left, bottom_right;

  // Attributes
  Point centre;
  int width, height;

  inline bool operator==(const Rect &rct) const {
    return top_left == rct.top_left && bottom_right == rct.bottom_right;
  }

  inline bool operator!=(const Rect &rct) const {
    return top_left != rct.top_left || bottom_right != rct.bottom_right;
  }

  Rect() = default;

  Rect(Point top_left_val, Point bottom_right_val)
      : top_left(top_left_val),
        bottom_right(bottom_right_val),
        centre(static_cast<int>(std::round((top_left_val + bottom_right_val).x / 2.0)),
               static_cast<int>(std::round((top_left_val + bottom_right_val).y / 2.0))),
        width((top_left_val - bottom_right_val).x),
        height((top_left_val - bottom_right_val).y) {}

  /// Get the Chebyshev distance to another rect.
  ///
  /// @param other - The rect to find the distance to.
  /// @return The Chebyshev distance between this rect and the given rect.
  [[nodiscard]] int get_distance_to(const Rect &other) const;

  /// Place the rect in the 2D grid.
  ///
  /// @param grid - The 2D grid which represents the dungeon.
  void place_rect(Grid &grid) const;
};

// ----- FUNCTIONS ------------------------------
/// Allows multiple hashes to be combined for a struct
///
/// @param seed - The seed for initialising the hasher.
/// @param v - The value to hash.
template<class T>
inline void hash_combine(size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<>
struct std::hash<Point> {
  size_t operator()(const Point &pnt) const {
    size_t res = 0;
    hash_combine(res, pnt.x);
    hash_combine(res, pnt.y);
    return res;
  }
};

template<>
struct std::hash<Rect> {
  size_t operator()(const Rect &rct) const {
    size_t res = 0;
    hash_combine(res, rct.top_left);
    hash_combine(res, rct.bottom_right);
    return res;
  }
};
