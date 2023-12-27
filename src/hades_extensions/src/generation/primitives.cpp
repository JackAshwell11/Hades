// Related header
#include "generation/primitives.hpp"

// ----- FUNCTIONS ------------------------------
void Grid::place_rect(const Rect &rect) const {
  // Place the walls
  for (int y = std::max(rect.top_left.y, 0); y < std::min(rect.bottom_right.y + 1, height); y++) {
    for (int x = std::max(rect.top_left.x, 0); x < std::min(rect.bottom_right.x + 1, width); x++) {
      if (get_value({x, y}) == TileType::Empty || get_value({x, y}) == TileType::Obstacle) {
        set_value({x, y}, TileType::Wall);
      }
    }
  }

  // Place the floors making sure the ranges are -1 in all directions since we don't want to overwrite the walls keeping
  // the player in, but we still want to overwrite walls that block the path for hallways
  for (int y = std::max(rect.top_left.y + 1, 1); y < std::min(rect.bottom_right.y, height - 1); y++) {
    for (int x = std::max(rect.top_left.x + 1, 1); x < std::min(rect.bottom_right.x, width - 1); x++) {
      set_value({x, y}, TileType::Floor);
    }
  }
}
