// Related header
#include "generation/bsp.hpp"

// Local headers
#include "generation/primitives.hpp"

namespace {
/// The maximum width to height ratio that is allowed.
constexpr double CONTAINER_RATIO{1.25};

/// The minimum size a container can be.
constexpr int MIN_CONTAINER_SIZE{5};

/// The minimum size a room can be.
constexpr int MIN_ROOM_SIZE{4};

/// The minimum width to height ratio that is allowed for a room.
constexpr double ROOM_RATIO{0.625};
}  // namespace

Leaf::Leaf(const Rect& container) : container{std::make_unique<Rect>(container)} {}

void Leaf::split(std::mt19937& random_generator) {  // NOLINT(misc-no-recursion)
  // Check if this leaf is already split or not
  if (left && right) {
    return;
  }

  // To determine which direction to split it, we have three options:
  //   1. Split vertically if the width is 25% larger than the height.
  //   2. Split horizontally if the height is 25% larger than the width.
  //   3. Split randomly if neither of the above is true.
  bool split_vertical;  // NOLINT(cppcoreguidelines-init-variables)
  if (container->width >= CONTAINER_RATIO * container->height) {
    split_vertical = true;
  } else if (container->height >= CONTAINER_RATIO * container->width) {
    split_vertical = false;
  } else {
    split_vertical = std::uniform_int_distribution{0, 1}(random_generator) == 1;
  }

  // Check if the container is too small to split
  const int max_size{split_vertical ? container->width - MIN_CONTAINER_SIZE : container->height - MIN_CONTAINER_SIZE};
  if (max_size <= MIN_CONTAINER_SIZE) {
    return;
  }

  // Determine the random split position to use ensuring that the containers are at least MIN_CONTAINER_SIZE wide
  const int pos{std::uniform_int_distribution{MIN_CONTAINER_SIZE, max_size}(random_generator)};
  const int split_pos{split_vertical ? container->top_left.x + pos : container->top_left.y + pos};

  // Generate the left and right leafs making sure that the containers do not include the split position
  if (split_vertical) {
    left = std::make_unique<Leaf>(Rect{{.x = container->top_left.x, .y = container->top_left.y},
                                       {.x = split_pos - 1, .y = container->bottom_right.y}});
    right = std::make_unique<Leaf>(Rect{{.x = split_pos + 1, .y = container->top_left.y},
                                        {.x = container->bottom_right.x, .y = container->bottom_right.y}});
  } else {
    left = std::make_unique<Leaf>(Rect{{.x = container->top_left.x, .y = container->top_left.y},
                                       {.x = container->bottom_right.x, .y = split_pos - 1}});
    right = std::make_unique<Leaf>(Rect{{.x = container->top_left.x, .y = split_pos + 1},
                                        {.x = container->bottom_right.x, .y = container->bottom_right.y}});
  }

  // Split the left and right leafs until they are too small to split
  left->split(random_generator);
  right->split(random_generator);
}

void Leaf::create_room(Grid& grid, std::mt19937& random_generator,  // NOLINT(misc-no-recursion)
                       std::vector<Position>& rooms) {
  // Check if this leaf is already split or not, if so, create rooms for the left and right leafs
  if (left && right) {
    left->create_room(grid, random_generator, rooms);
    right->create_room(grid, random_generator, rooms);
    return;
  }

  // Check if this leaf is too small to create a room. If the leaf has been split correctly, this should never happen
  if (container->width < MIN_ROOM_SIZE || container->height < MIN_ROOM_SIZE) {
    return;
  }

  // Determine the width and height of the room making sure it is at least MIN_ROOM_SIZE wide
  const int width{std::uniform_int_distribution{MIN_ROOM_SIZE, container->width}(random_generator)};
  const int height{std::uniform_int_distribution{MIN_ROOM_SIZE, container->height}(random_generator)};

  // Determine the top left position of the new room based on the width and height
  const int x_pos{
      std::uniform_int_distribution{container->top_left.x, container->bottom_right.x - width}(random_generator)};
  const int y_pos{
      std::uniform_int_distribution{container->top_left.y, container->bottom_right.y - height}(random_generator)};

  // Create the room rect and check its width to height ratio so oddly shaped rooms can be avoided
  const Rect rect{{.x = x_pos, .y = y_pos}, {.x = x_pos + width - 1, .y = y_pos + height - 1}};
  if (static_cast<double>(std::min(rect.width, rect.height)) / std::max(rect.width, rect.height) < ROOM_RATIO) {
    // Since MIN_ROOM_SIZE ensures the random generator will always raise an exception if a leaf is too small, a valid
    // room will always be created, so we can just keep trying
    create_room(grid, random_generator, rooms);
    return;
  }

  // Place the rect in the 2D grid then save it in the leaf and the rooms vector
  grid.place_rect(rect);
  room = std::make_unique<Rect>(rect);
  rooms.push_back(rect.centre);
}
