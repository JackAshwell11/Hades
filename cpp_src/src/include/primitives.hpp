// Ensure this file is only included once
#pragma once

// Std includes
#include <cmath>
#include <vector>

// ----- ENUMS ------------------------------
/// Stores the different types of tiles in the game map.
enum class TileType {
  DebugWall,
  Empty,
  Floor,
  Wall,
  Obstacle,
  Player,
  HealthPotion,
  ArmourPotion,
  HealthBoostPotion,
  ArmourBoostPotion,
  SpeedBoostPotion,
  FireRateBoostPotion,
};

// ----- STRUCTURES ------------------------------
/// Represents a point in the grid.
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

  /// Default constructor for a Point object. This should not be used.
  Point() = default;

  /// Construct a Point object.
  ///
  /// Parameters
  /// ----------
  /// x - The x position.
  /// y - The y position.
  ///
  /// Returns
  /// -------
  /// A Point object.
  Point(int x_val, int y_val) {
    x = x_val;
    y = y_val;
  }
};

/// Represents a 2D grid with a set width and height through a 1D vector.
///
/// Attributes
/// ----------
/// width - The width of the 2D grid.
/// height - The height of the 2D grid.
struct Grid {
  // Parameters
  int width{}, height{};

  // Attributes
  std::vector<TileType> grid{};

  /// Default constructor for a Grid object. This should not be used.
  Grid() = default;

  /// Construct a Grid object.
  ///
  /// Parameters
  /// ----------
  /// width - The width of the 2D grid.
  /// height - The height of the 2D grid.
  ///
  /// Returns
  /// -------
  /// A Grid object.
  Grid(int width_val, int height_val) {
    width = width_val;
    height = height_val;
    grid = std::vector<TileType>(width * height, TileType::Empty);
  }

  /// Convert a 2D grid position to a 1D grid position.
  ///
  /// Parameters
  /// ----------
  /// pos - The position to convert.
  ///
  /// Throws
  /// ------
  /// std::out_of_range
  ///     Position must be within range.
  ///
  /// Returns
  /// -------
  /// The 1D grid position.
  [[nodiscard]] int convert_position(const Point &pos) const;

  /// Get a value in the 2D grid from a given position.
  ///
  /// Parameters
  /// ----------
  /// pos - The position to get the value for.
  ///
  /// Throws
  /// ------
  /// std::out_of_range
  ///     Position must be within range.
  ///
  /// Returns
  /// -------
  /// The value at the given position.
  [[nodiscard]] TileType get_value(const Point &pos) const;

  /// Set a value in the 2D grid from a given position.
  ///
  /// Parameters
  /// ----------
  /// pos - The position to set.
  ///
  /// Throws
  /// ------
  /// std::out_of_range
  ///     Position must be within range.
  void set_value(const Point &pos, TileType target);
};

/// Represents a rectangle of any size useful for the interacting with the 2D
/// grid.
///
/// When creating a container, the split wall is included in the rect size,
/// whereas, rooms don't so MIN_CONTAINER_SIZE must be bigger than
/// MIN_ROOM_SIZE.
///
/// Attributes
/// ----------
/// center - The center position of the rect.
/// width - The width of the rect.
/// height - The height of the rect.
struct Rect {
  // Parameters
  Point top_left{}, bottom_right{};

  // Attributes
  Point center{};
  int width{}, height{};

  inline bool operator==(const Rect rct) const {
    return top_left == rct.top_left && bottom_right == rct.bottom_right;
  }

  inline bool operator!=(const Rect rct) const {
    return top_left != rct.top_left || bottom_right != rct.bottom_right;
  }

  /// Default constructor for a Rect object. This should not be used.
  Rect() = default;

  /// Construct a Rect object.
  ///
  /// Parameters
  /// ----------
  /// top-left - The top-left position.
  /// bottom-right - The bottom-right position.
  ///
  /// Returns
  /// -------
  /// A Rect object.
  Rect(Point top_left_val, Point bottom_right_val) {
    Point sum = top_left_val + bottom_right_val;
    Point diff = top_left_val - bottom_right_val;
    top_left = top_left_val;
    bottom_right = bottom_right_val;
    center = Point((int) round(((double) sum.x) / 2.0),
                   (int) round(((double) sum.y) / 2.0));
    width = diff.x;
    height = diff.y;
  }

  /// Get the Chebyshev distance to another rect.
  ///
  /// Parameters
  /// ----------
  /// other - The rect to find the distance to.
  ///
  /// Returns
  /// -------
  /// The Chebyshev distance between this rect and the given rect.
  [[nodiscard]] int get_distance_to(const Rect &other) const;

  /// Place the rect in the 2D grid.
  ///
  /// Parameters
  /// ----------
  /// grid - The 2D grid which represents the dungeon.
  void place_rect(Grid &grid) const;
};

// ----- FUNCTIONS ------------------------------
/// Allows multiple hashes to be combined for a struct
template<class T>
inline void hash_combine(size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/// Allows the Point struct to be hashed in a map
template<>
struct std::hash<Point> {
  size_t operator()(const Point &pnt) const {
    size_t res = 0;
    hash_combine(res, pnt.x);
    hash_combine(res, pnt.y);
    return res;
  }
};

/// Allows the Rect struct to be hashed in a map
template<>
struct std::hash<Rect> {
  size_t operator()(const Rect &rct) const {
    size_t res = 0;
    hash_combine(res, rct.top_left);
    hash_combine(res, rct.bottom_right);
    return res;
  }
};
