// Custom includes
#include "primitives.hpp"

#include <iostream>

// ----- CONSTANTS ------------------------------
// Defines constants for hallway and entity generation
const TileType REPLACEABLE_TILES[3] = {TileType::Empty, TileType::Obstacle,
                                       TileType::DebugWall};

// ----- FUNCTIONS ------------------------------
int Rect::get_distance_to(Rect &other) const {
  return std::max(abs(center.x - other.center.x),
                  abs(center.y - other.center.y));
}

void Rect::place_rect(Grid &grid) const {
  // Place the walls
  for (int y = std::max(top_left.y, 0);
       y < std::min(bottom_right.y + 1, grid.height); y++) {
    for (int x = std::max(top_left.x, 0);
         x < std::min(bottom_right.x + 1, grid.width); x++) {
      if (std::count(std::begin(REPLACEABLE_TILES),
                     std::end(REPLACEABLE_TILES), grid.get_value(x, y)) > 0) {
        grid.set_value(x, y, TileType::Wall);
      }
    }
  }

  // Place the floors. The ranges must be -1 in all directions since we don't
  // want to overwrite the walls keeping the player in, but we still want to
  // overwrite walls that block the path for hallways
  for (int y = std::max(top_left.y + 1, 1);
       y < std::min(bottom_right.y, grid.height - 1); y++) {
    for (int x = std::max(top_left.x + 1, 1);
         x < std::min(bottom_right.x, grid.width - 1); x++) {
      grid.set_value(x, y, TileType::Floor);
    }
  }
}

TileType Grid::get_value(int x, int y) const {
  // Convert the 2D position to 1D and return the value at that position
  return grid.at(width * y + x);
}

void Grid::set_value(int x, int y, TileType target) {
  // Convert the 2D position to 1D and set the value at that position to target
  if (x < 0 || x >= width || y < 0 || y >= height) {
    throw std::out_of_range("Position must be within range");
  }
  grid[width * y + x] = target;
}
