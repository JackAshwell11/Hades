// Related header
#include "generation/primitives.hpp"

// Std headers
#include <array>
#include <stdexcept>

namespace {
/// The offsets for the intercardinal directions.
constexpr std::array<Position, 8> INTERCARDINAL_OFFSETS{{{.x = -1, .y = -1},  // North-West
                                                         {.x = 0, .y = -1},   // North
                                                         {.x = 1, .y = -1},   // North-East
                                                         {.x = -1, .y = 0},   // West
                                                         {.x = 1, .y = 0},    // East
                                                         {.x = -1, .y = 1},   // South-West
                                                         {.x = 0, .y = 1},    // South
                                                         {.x = 1, .y = 1}}};  // South-East
}  // namespace

auto Position::get_distance_to(const Position &other) const -> int {
  return std::max(abs(x - other.x), abs(y - other.y));
}

Rect::Rect(const Position &top_left, const Position &bottom_right)
    : top_left(top_left),
      bottom_right(bottom_right),
      centre({.x = (top_left.x + bottom_right.x + 1) / 2, .y = (top_left.y + bottom_right.y + 1) / 2}),
      width((top_left - bottom_right).x),
      height((top_left - bottom_right).y) {}

Grid::Grid(const int width, const int height)
    : width(width), height(height), grid(std::vector(width * height, GameObjectType::Empty)) {}

auto Grid::is_position_within(const Position &position) const -> bool {
  return position.x >= 0 && position.x < width && position.y >= 0 && position.y < height;
}

auto Grid::convert_position(const int position) const -> Position {
  if (position < 0 || position >= width * height) {
    throw std::out_of_range("Position not within the grid.");
  }
  return {.x = position % width, .y = position / width};
}

auto Grid::convert_position(const Position &position) const -> int {
  if (!is_position_within(position)) {
    throw std::out_of_range("Position not within the grid.");
  }
  return (position.y * width) + position.x;
}

auto Grid::get_value(const Position &position) const -> GameObjectType { return grid.at(convert_position(position)); }

void Grid::set_value(const Position &position, const GameObjectType target) {
  grid.at(convert_position(position)) = target;
}

auto Grid::get_neighbours(const Position &position) const -> std::vector<Position> {
  std::vector<Position> neighbours;
  for (const Position &offset : INTERCARDINAL_OFFSETS) {
    if (const Position neighbour{position + offset}; is_position_within(neighbour)) {
      neighbours.emplace_back(neighbour);
    }
  }
  return neighbours;
}

void Grid::place_rect(const Rect &rect) {
  for (auto y{std::max(rect.top_left.y, 0)}; y < std::min(rect.bottom_right.y + 1, height); y++) {
    for (auto x{std::max(rect.top_left.x, 0)}; x < std::min(rect.bottom_right.x + 1, width); x++) {
      set_value({.x = x, .y = y}, GameObjectType::Floor);
    }
  }
}
