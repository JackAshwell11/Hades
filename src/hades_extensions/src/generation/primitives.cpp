// Related header
#include "generation/primitives.hpp"

// ----- FUNCTIONS ------------------------------
void Rect::place_rect(Grid &grid) const {
  // Place the walls
  for (int y = std::max(top_left.y, 0); y < std::min(bottom_right.y + 1, grid.height); y++) {
    for (int x = std::max(top_left.x, 0); x < std::min(bottom_right.x + 1, grid.width); x++) {
      if (grid.get_value({x, y}) == TileType::Empty || grid.get_value({x, y}) == TileType::Obstacle) {
        grid.set_value({x, y}, TileType::Wall);
      }
    }
  }

  // Place the floors making sure the ranges are -1 in all directions since we don't want to overwrite the walls keeping
  // the player in, but we still want to overwrite walls that block the path for hallways
  for (int y = std::max(top_left.y + 1, 1); y < std::min(bottom_right.y, grid.height - 1); y++) {
    for (int x = std::max(top_left.x + 1, 1); x < std::min(bottom_right.x, grid.width - 1); x++) {
      grid.set_value({x, y}, TileType::Floor);
    }
  }
}
