// Std includes
#include <random>

// Custom includes
#include "bsp.hpp"

// ----- CONSTANTS ------------------------------
#define CONTAINER_RATIO 1.25
#define MIN_CONTAINER_SIZE 5
#define MIN_ROOM_SIZE 4
#define ROOM_RATIO 0.625

// ----- FUNCTIONS ------------------------------
bool split(Leaf &leaf, std::mt19937 &random_generator) {
  // Check if this leaf is already split or not
  if (leaf.left && leaf.right) {
    return false;
  }

  // To determine the direction of split, we test if the width is 25% larger
  // than the height, if so we split vertically. However, if the height is 25%
  // larger than the width, we split horizontally. Otherwise, we split randomly
  bool split_vertical;
  if (leaf.container->width > leaf.container->height
      && (leaf.container->width >= CONTAINER_RATIO * leaf.container->height)) {
    split_vertical = true;
  } else if (leaf.container->height > leaf.container->width
      && (leaf.container->height >= CONTAINER_RATIO * leaf.container->width)) {
    split_vertical = false;
  } else {
    std::uniform_int_distribution<> split_vertical_distribution{0, 1};
    split_vertical = split_vertical_distribution(random_generator);
  }

  // To determine the range of values that we could split on, we need to find
  // out if the container is too small. Once we've done that, we can use the
  // x1, y1, x2, and y2 coordinates to specify the range of values
  int max_size =
      (split_vertical) ? leaf.container->width - MIN_CONTAINER_SIZE : leaf.container->height - MIN_CONTAINER_SIZE;
  if (max_size <= MIN_CONTAINER_SIZE) {
    // Container too small to split
    return false;
  }

  // Create the split position. This ensures that there will be
  // MIN_CONTAINER_SIZE on each side
  std::uniform_int_distribution<> pos_distribution{MIN_CONTAINER_SIZE, max_size};
  int pos = pos_distribution(random_generator);
  int split_pos = (split_vertical) ? leaf.container->top_left.x + pos : leaf.container->top_left.y + pos;

  // Split the container
  if (split_vertical) {
    leaf.left = std::make_unique<Leaf>(Rect{{leaf.container->top_left.x, leaf.container->top_left.y},
                                            {split_pos - 1, leaf.container->bottom_right.y}});
    leaf.right = std::make_unique<Leaf>(Rect{{split_pos + 1, leaf.container->top_left.y},
                                             {leaf.container->bottom_right.x, leaf.container->bottom_right.y}});
  } else {
    leaf.left = std::make_unique<Leaf>(Rect{{leaf.container->top_left.x, leaf.container->top_left.y},
                                            {leaf.container->bottom_right.x, split_pos - 1}});
    leaf.right = std::make_unique<Leaf>(Rect{{leaf.container->top_left.x, split_pos + 1},
                                             {leaf.container->bottom_right.x, leaf.container->bottom_right.y}});
  }

  // Successful split
  return true;
}

bool create_room(Leaf &leaf, Grid &grid, std::mt19937 &random_generator) {
  // Test if this container is already split or not. If it is, we do not want
  // to create a room inside it otherwise it will overwrite other rooms
  if (leaf.left && leaf.right) {
    return false;
  }

  // Pick a random width and height making sure it is at least min_room_size
  // but doesn't exceed the container
  std::uniform_int_distribution<> width_distribution{MIN_ROOM_SIZE, leaf.container->width};
  int width = width_distribution(random_generator);
  std::uniform_int_distribution<> height_distribution{MIN_ROOM_SIZE, leaf.container->height};
  int height = height_distribution(random_generator);

  // Use the width and height to find a suitable x and y position which can
  // create the room
  std::uniform_int_distribution<>
      x_pos_distribution{leaf.container->top_left.x, leaf.container->bottom_right.x - width};
  int x_pos = x_pos_distribution(random_generator);
  std::uniform_int_distribution<>
      y_pos_distribution{leaf.container->top_left.y, leaf.container->bottom_right.y - height};
  int y_pos = y_pos_distribution(random_generator);

  // Create the room rect and test if its width to height ratio will make an
  // oddly-shaped room
  Rect rect{{x_pos, y_pos}, {x_pos + width - 1, y_pos + height - 1}};
  if ((static_cast<double>(std::min(rect.width, rect.height)) / std::max(rect.width, rect.height)) < ROOM_RATIO) {
    return false;
  }

  // Width to height ratio is fine so place the rect in the 2D grid and store
  // it
  rect.place_rect(grid);
  leaf.room = std::make_unique<Rect>(rect);

  // Successful room creation
  return true;
}
