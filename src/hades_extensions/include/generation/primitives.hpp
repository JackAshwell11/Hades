// Ensure this file is only included once
#pragma once

// Std headers
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

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

/// Represents a 2D position.
struct Position {
  /// The equality operator.
  auto operator==(const Position &position) const -> bool { return x == position.x && y == position.y; }

  /// The inequality operator.
  auto operator!=(const Position &position) const -> bool { return x != position.x || y != position.y; }

  /// The addition operator.
  auto operator+(const Position &position) const -> Position { return {.x = x + position.x, .y = y + position.y}; }

  /// The subtraction operator.
  auto operator-(const Position &position) const -> Position {
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
  [[nodiscard]] auto get_distance_to(const Position &other) const -> int {
    return std::max(abs(x - other.x), abs(y - other.y));
  }
};

/// Represents a rectangle in 2D space.
struct Rect {
  auto operator==(const Rect &rect) const -> bool {
    return top_left == rect.top_left && bottom_right == rect.bottom_right;
  }

  auto operator!=(const Rect &rect) const -> bool {
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
        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        centre({.x = static_cast<int>(std::round((top_left + bottom_right).x / 2.0)),
                .y = static_cast<int>(std::round((top_left + bottom_right).y / 2.0))}),
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        width((top_left - bottom_right).x),
        height((top_left - bottom_right).y) {}
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
  static constexpr std::array INTERCARDINAL_OFFSETS{Position{.x = -1, .y = -1},  // North-west
                                                    Position{.x = 0, .y = -1},   // North
                                                    Position{.x = 1, .y = -1},   // North-east
                                                    Position{.x = -1, .y = 0},   // West
                                                    Position{.x = 1, .y = 0},    // East
                                                    Position{.x = -1, .y = 1},   // South-west
                                                    Position{.x = 0, .y = 1},    // South
                                                    Position{.x = 1, .y = 1}};   // South-east

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
  [[nodiscard]] auto is_position_within(const Position &position) const -> bool {
    return position.x >= 0 && position.x < width && position.y >= 0 && position.y < height;
  }

  /// Convert a 1D grid position to a 2D grid position.
  ///
  /// @param position - The position to convert.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  /// @return The 2D grid position.
  [[nodiscard]] auto convert_position(const int position) const -> Position {
    if (position < 0 || position >= width * height) {
      throw std::out_of_range("Position not within the grid.");
    }
    return {.x = position % width, .y = position / width};
  }

  /// Convert a 2D grid position to a 1D grid position.
  ///
  /// @param position - The position to convert.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  /// @return The 1D grid position.
  [[nodiscard]] auto convert_position(const Position &position) const -> int {
    if (!is_position_within(position)) {
      throw std::out_of_range("Position not within the grid.");
    }
    return (width * position.y) + position.x;
  }

  /// Get a value in the 2D grid from a given position.
  ///
  /// @param position - The position to get the value for.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  /// @return The value at the given position.
  [[nodiscard]] auto get_value(const Position &position) const -> TileType {
    return grid->at(convert_position(position));
  }

  /// Set a value in the 2D grid from a given position.
  ///
  /// @param position - The position to set.
  /// @param target - The value to set at the given position.
  /// @throws std::out_of_range - If the position is not within the 2D grid.
  void set_value(const Position &position, const TileType target) const {
    grid->at(convert_position(position)) = target;
  }

  /// Get the neighbours of a given position.
  ///
  /// @param position - The position to get the neighbours for.
  /// @return The neighbours of the given position.
  [[nodiscard]] auto get_neighbours(const Position &position) const -> std::vector<Position> {
    std::vector<Position> neighbours;
    for (const Position &offset : INTERCARDINAL_OFFSETS) {
      if (const Position neighbour{position + offset}; is_position_within(neighbour)) {
        neighbours.emplace_back(neighbour);
      }
    }
    return neighbours;
  }

  /// Place a rect in the 2D grid.
  ///
  /// @param rect - The rect to place in the 2D grid.
  void place_rect(const Rect &rect) const {
    for (int y{std::max(rect.top_left.y, 0)}; y < std::min(rect.bottom_right.y + 1, height); y++) {
      for (int x{std::max(rect.top_left.x, 0)}; x < std::min(rect.bottom_right.x + 1, width); x++) {
        set_value({.x = x, .y = y}, TileType::Floor);
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
struct std::hash<Position> {
  auto operator()(const Position &position) const noexcept -> std::size_t {
    std::size_t res{0};
    hash_combine(res, position.x);
    hash_combine(res, position.y);
    return res;
  }
};
