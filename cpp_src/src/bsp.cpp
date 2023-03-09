// Custom includes
#include "bsp.hpp"
#include "primitives.hpp"

// ----- CONSTANTS ------------------------------
#define CONTAINER_RATIO 1.25
#define MIN_CONTAINER_SIZE 5
#define MIN_ROOM_SIZE 4
#define ROOM_RATIO 0.625

// ----- FUNCTIONS ------------------------------
bool Leaf::split(std::vector<std::vector<TileType>> &grid,
                 std::mt19937 &random_generator, bool debug_game) {
  // Check if this leaf is already split or not
  if (left && right) {
    return false;
  }

  // To determine the direction of split, we test if the width is 25% larger
  // than the height, if so we split vertically. However, if the height is 25%
  // larger than the width, we split horizontally. Otherwise, we split
  // randomly
  bool split_vertical;
  if ((container.width > container.height) &&
      (((double) container.width / container.height) >= CONTAINER_RATIO)) {
    split_vertical = true;
  } else if ((container.height > container.width) &&
      (((double) container.height / container.width) >= CONTAINER_RATIO)) {
    split_vertical = false;
  } else {
    std::uniform_int_distribution<> split_vertical_distribution(0, 1);
    split_vertical = split_vertical_distribution(random_generator);
  }

  // To determine the range of values that we could split on, we need to find
  // out if the container is too small. Once we've done that, we can use the
  // x1, y1, x2 and y2 coordinates to specify the range of values
  int max_size = (split_vertical) ? container.width - MIN_CONTAINER_SIZE
                                  : container.height - MIN_CONTAINER_SIZE;
  if (max_size <= MIN_CONTAINER_SIZE) {
    // Container too small to split
    return false;
  }

  // Create the split position. This ensures that there will be
  // MIN_CONTAINER_SIZE on each side
  std::uniform_int_distribution<> pos_distribution(MIN_CONTAINER_SIZE,
                                                   max_size);
  int pos = pos_distribution(random_generator);

  // Split the container
  if (split_vertical) {
    // Split vertically making sure to adjust pos, so it can be within range
    // of the actual container
    pos += container.top_left.x;
    if (debug_game) {
      for (int y = container.top_left.y; y < container.bottom_right.y + 1; y++) {
        grid[y][pos] = TileType::DebugWall;
      }
    }

    // Create the child leafs
    left = new Leaf{
        {
            {container.top_left.x, container.top_left.y},
            {pos - 1, container.bottom_right.y},
        }
    };
    right = new Leaf{
        {
            {pos + 1, container.top_left.y},
            {container.bottom_right.x, container.bottom_right.y},
        }
    };
  } else {
    // Split horizontally making sure to adjust pos, so it can be within range
    // of the actual container
    pos += container.top_left.y;
    if (debug_game) {
      for (int x = container.top_left.x; x < container.bottom_right.x + 1; x++) {
        grid[pos][x] = TileType::DebugWall;
      }
    }

    // Create the child leafs
    left = new Leaf{
        {
            {container.top_left.x, container.top_left.y},
            {container.bottom_right.x, pos - 1},
        }
    };
    right = new Leaf{
        {
            {container.top_left.x, pos + 1},
            {container.bottom_right.x, container.bottom_right.y},
        }
    };
  }

  // Successful split
  return true;
}

bool Leaf::create_room(std::vector<std::vector<TileType>> &grid,
                       std::mt19937 &random_generator) {
  // Test if this container is already split or not. If it is, we do not want
  // to create a room inside it otherwise it will overwrite other rooms
  if (left && right) {
    return false;
  }

  // Pick a random width and height making sure it is at least min_room_size
  // but doesn't exceed the container
  std::uniform_int_distribution<> width_distribution(MIN_ROOM_SIZE, container.width);
  int width = width_distribution(random_generator);
  std::uniform_int_distribution<> height_distribution(MIN_ROOM_SIZE, container.height);
  int height = height_distribution(random_generator);

  // Use the width and height to find a suitable x and y position which can
  // create the room
  std::uniform_int_distribution<> x_pos_distribution(container.top_left.x, container.bottom_right.x - width);
  int x_pos = x_pos_distribution(random_generator);
  std::uniform_int_distribution<> y_pos_distribution(container.top_left.y, container.bottom_right.y - height);
  int y_pos = y_pos_distribution(random_generator);

  // Create the room rect and test if its width to height ratio will make an
  // oddly-shaped room
  Rect rect = {{x_pos, y_pos}, {x_pos + width - 1, y_pos + height - 1}};
  if ((((double) std::min(rect.width, rect.height)) /
      ((double) std::min(rect.width, rect.height))) < ROOM_RATIO) {
    return false;
  }

  // Width to height ratio is fine so place the rect in the 2D grid and store
  // it
  rect.place_rect(grid);
  room = &rect;

  // Successful room creation
  return true;
}
