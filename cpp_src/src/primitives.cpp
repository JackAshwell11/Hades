// Custom includes
#include "primitives.hpp"

// ----- STRUCTURES ------------------------------
std::pair<int, int> Point::sum(Point &other) const {
  return std::make_pair(x + other.x, y + other.y);
}

std::pair<int, int> Point::abs_diff(Point &other) const {
  return std::make_pair(abs(x - other.x), abs(y - other.y));
}

int Rect::get_distance_to(Rect &other) const {
  return std::max(abs(center.x - other.center.x),
                  abs(center.y - other.center.y));
}

void Rect::place_rect(std::vector<std::vector<TileType>> &grid) const {
  // Get the width and height of the grid
  int grid_height = (int) grid.size();
  int grid_width = (int) grid[0].size();

  // Place the walls
  for (int y = std::max(top_left.y, 0);
       y < std::min(bottom_right.y + 1, grid_height); y++) {
    for (int x = std::max(top_left.x, 0);
         x < std::min(bottom_right.x + 1, grid_width); x++) {
      if (std::count(std::begin(REPLACEABLE_TILES),
                     std::end(REPLACEABLE_TILES), grid[y][x]) > 0) {
        grid[y][x] = TileType::Wall;
      }
    }
  }

  // Place the floors. The ranges must be -1 in all directions since we don't
  // want to overwrite the walls keeping the player in, but we still want to
  // overwrite walls that block the path for hallways
  for (int y = std::max(top_left.y + 1, 1);
       y < std::min(bottom_right.y, grid_height - 1); y++) {
    for (int x = std::max(top_left.x + 1, 1);
         x < std::min(bottom_right.x, grid_width - 1); x++) {
      grid[y][x] = TileType::Floor;
    }
  }
}
