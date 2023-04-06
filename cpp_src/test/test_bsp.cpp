// External includes
#include "gtest/gtest.h"

// Custom includes
#include "bsp.hpp"
#include "fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestBspSplitCorrect) {
  // Split the bsp normally
  leaf.split(grid, random_generator, false);

  // Test if two child leafs are created
  ASSERT_TRUE(leaf.left != nullptr);
  ASSERT_TRUE(leaf.right != nullptr);

  // Test if the child leafs border each other. Since the leaf will always split
  // vertically on the first iteration, we know the difference between the
  // bottom right and top left points of the left and right leafs
  Point diff = {2, 19};
  ASSERT_EQ(leaf.left->container.bottom_right - leaf.right->container.top_left, diff);
}

TEST_F(Fixtures, TestBspSplitDebug) {
  // Split the grid in debug mode and then find the row/column where the split occurred
  leaf.split(grid, random_generator, true);
  auto positions = std::find(grid.grid.begin(), grid.grid.end(), TileType::DebugWall);
  ASSERT_TRUE(positions != grid.grid.end());

  // Get the grid position of the split
  int debug_pos_index = (int) (positions - grid.grid.begin());
  Point debug_pos = {debug_pos_index % grid.width, debug_pos_index / grid.width};

  // Make sure the split runs the entire length of the grid. We need to
  // determine if the split was vertical or horizontal first however
  if (debug_pos.x > 0) {
    for (int y = 0; y < grid.width; y++) {
      ASSERT_EQ(grid.get_value({debug_pos.x, y}), TileType::DebugWall);
    }
  } else if (debug_pos.y > 0) {
    for (int x = 0; x < grid.height; x++) {
      ASSERT_EQ(grid.get_value({x, debug_pos.y}), TileType::DebugWall);
    }
  }
}

TEST_F(Fixtures, TestBspSplitSmallWidthHeight) {
  // Make sure we test what happens if the container's width and height are both
  // less than MIN_CONTAINER_SIZE
  leaf.container = Rect{Point{-1, -1}, Point{-1, -1}};
  ASSERT_FALSE(leaf.split(grid, random_generator, false));
}

TEST_F(Fixtures, TestBspCreateRoomChildLeaf) {
  // Repeat until a room is created since the ratio may be wrong sometimes then
  // test that the room is not null
  while (!leaf.create_room(grid, random_generator)) {}
  ASSERT_TRUE(leaf.room != nullptr);
}

TEST_F(Fixtures, TestBspCreateRoomNotNullLeftRight) {
  // Test what happens if the leaf and right leafs are not null
  Leaf temp_leaf = Leaf{Rect{Point{0, 0}, Point{0, 0}}};
  leaf.left = leaf.right = &temp_leaf;
  ASSERT_FALSE(leaf.create_room(grid, random_generator));
}
