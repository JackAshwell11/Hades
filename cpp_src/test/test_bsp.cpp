// External includes
#include "gtest/gtest.h"

// Custom includes
#include "bsp.hpp"
#include "fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestBspSplitCorrect) {
  // Split the bsp normally
  leaf.split(grid, random_generator, false);

  // Calculate the difference between the bottom right point of the left leaf
  // and the top right point of the right leaf, so we can determine it's split
  // direction
  Point leaf_diff = leaf.left->container.bottom_right - leaf.right->container.top_left,
      target_diff = (leaf_diff.x == 2) ? Point{2, 19} : Point{19, 2};

  // Test if the child leafs border each other
  ASSERT_EQ(leaf_diff, target_diff);
}

TEST_F(Fixtures, TestBspSplitDebug) {
  // Split the grid in debug mode
  leaf.split(grid, random_generator, true);

  // Find the row/column where the split occurred and get the grid position of the split
  int debug_pos_index = (int) (std::find(grid.grid.begin(), grid.grid.end(), TileType::DebugWall) - grid.grid.begin());
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
