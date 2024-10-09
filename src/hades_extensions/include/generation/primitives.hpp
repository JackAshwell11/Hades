// Ensure this file is only included once
#pragma once

// Std headers
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

// Local headers
#include "vect.hpp"

/// Stores the different types of tiles in the game map.
enum struct TileType : std::uint8_t {
  Empty,
  Floor,
  Wall,
  Obstacle,
  Goal,
  Player,
  HealthPotion,
  Chest,
};

/// Get the Chebyshev distance between two vectors.
///
/// @param lhs - The left hand side vector.
/// @param rhs - The right hand side vector.
/// @return The Chebyshev distance between the two vectors.
[[nodiscard]] inline auto get_distance_to(const cpVect lhs, const cpVect rhs) -> double {
  return std::max(abs(lhs.x - rhs.x), abs(lhs.y - rhs.y));
}

/// Represents a rectangle in 2D space.
struct Rect {
  auto operator==(const Rect &rect) const -> bool {
    return ((top_left == rect.top_left) != 0U) && ((bottom_right == rect.bottom_right) != 0U);
  }

  auto operator!=(const Rect &rect) const -> bool {
    return top_left != rect.top_left || bottom_right != rect.bottom_right;
  }

  /// The top left vector of the rect.
  cpVect top_left;

  /// The bottom right vector of the rect.
  cpVect bottom_right;

  /// The centre vector of the rect.
  cpVect centre;

  /// The width of the rect.
  int width;

  /// The height of the rect.
  int height;

  /// Initialise the object.
  ///
  /// @param top_left - The top left vector of the rect.
  /// @param bottom_right - The bottom right vector of the rect.
  Rect(const cpVect &top_left, const cpVect &bottom_right)
      : top_left(top_left),
        bottom_right(bottom_right),
        centre(cpvlerp(top_left, bottom_right, 0.5)),
        width(static_cast<int>(std::round(top_left.x - bottom_right.x))),
        height(static_cast<int>(std::round(top_left.y - bottom_right.y))) {}
};

/// Represents a 2D grid with a set width and height through a 1D vector.
struct Grid {
  /// The width of the 2D grid.
  int width;

  /// The height of the 2D grid.
  int height;

  /// The vector which represents the 2D grid.
  std::unique_ptr<std::vector<TileType>> grid;

  /// The offsets for the intercardinal directions.
  static constexpr std::array INTERCARDINAL_OFFSETS{cpVect{-1, -1},  // North-west
                                                    cpVect{0, -1},   // North
                                                    cpVect{1, -1},   // North-east
                                                    cpVect{-1, 0},   // West
                                                    cpVect{1, 0},    // East
                                                    cpVect{-1, 1},   // South-west
                                                    cpVect{0, 1},    // South
                                                    cpVect{1, 1}};   // South-east

  /// Initialise the object.
  ///
  /// @param width - The width of the 2D grid.
  /// @param height - The height of the 2D grid.
  Grid(const int width, const int height)
      : width(width), height(height), grid(std::make_unique<std::vector<TileType>>(width * height, TileType::Empty)) {}

  /// Check if a position is within the 2D grid.
  ///
  /// @param position - The position to check for.
  /// @return Whether the position is within the 2D grid or not.
  [[nodiscard]] auto is_position_within(const cpVect &position) const -> bool {
    return position.x >= 0 && position.x < width && position.y >= 0 && position.y < height;
  }

  /// Convert a 1D grid position to a 2D grid position.
  ///
  /// @param position - The position to convert.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  /// @return The 2D grid position.
  [[nodiscard]] auto convert_position(const int position) const -> cpVect {
    if (position < 0 || position >= width * height) {
      throw std::out_of_range("cpVect not within the grid.");
    }
    return cpv(position % width, position / width);
  }

  /// Convert a 2D grid position to a 1D grid position.
  ///
  /// @param position - The position to convert.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  /// @return The 1D grid position.
  [[nodiscard]] auto convert_position(const cpVect &position) const -> int {
    if (!is_position_within(position)) {
      throw std::out_of_range("cpVect not within the grid.");
    }
    return (width * position.y) + position.x;
  }

  /// Get a value in the 2D grid from a given position.
  ///
  /// @param position - The position to get the value for.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  /// @return The value at the given position.
  [[nodiscard]] auto get_value(const cpVect &position) const -> TileType {
    return grid->at(convert_position(position));
  }

  /// Set a value in the 2D grid from a given position.
  ///
  /// @param position - The position to set.
  /// @param target - The value to set at the given position.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  void set_value(const cpVect &position, const TileType target) const { grid->at(convert_position(position)) = target; }

  /// Get the neighbours of a given position.
  ///
  /// @param position - The position to get the neighbours for.
  /// @return The neighbours of the given position.
  [[nodiscard]] auto get_neighbours(const cpVect &position) const -> std::vector<cpVect> {
    std::vector<cpVect> neighbours;
    for (const cpVect &offset : INTERCARDINAL_OFFSETS) {
      if (const cpVect neighbour{position + offset}; is_position_within(neighbour)) {
        neighbours.emplace_back(neighbour);
      }
    }
    return neighbours;
  }

  /// Place a rect in the 2D grid.
  ///
  /// @details It is the responsibility of the caller to ensure that the rect fits in the grid.
  /// @param rect - The rect to place in the 2D grid.
  void place_rect(const Rect &rect) const {
    // TODO: Improve this
    for (int y{std::max(static_cast<int>(rect.top_left.y), 0)};
         y < std::min(static_cast<int>(rect.bottom_right.y) + 1, height); y++) {
      for (int x{std::max(static_cast<int>(rect.top_left.x), 0)};
           x < std::min(static_cast<int>(rect.bottom_right.x) + 1, width); x++) {
        set_value(cpv(x, y), TileType::Floor);
      }
    }
  }
};

/// Allows multiple hashes to be combined for a struct
///
/// @tparam T - The type of the value to hash.
/// @param seed - The seed for initialising the hasher.
/// @param value - The value to hash.
template <typename T>
void hash_combine(std::size_t &seed, const T &value) {
  const std::hash<T> hasher;
  seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  // NOLINT
}

template <>
struct std::hash<cpVect> {
  auto operator()(const cpVect &position) const noexcept -> std::size_t {
    std::size_t res{0};
    hash_combine(res, position.x);
    hash_combine(res, position.y);
    return res;
  }
};
