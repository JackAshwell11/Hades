// External includes
#include "gtest/gtest.h"

// Custom includes
#include "bsp.hpp"
#include "fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestBspSplitValid) {
  // Test if the bsp is split correctly
  leaf.split(grid, random_generator, false);
  Leaf left_result = Leaf{Rect{Point{0, 0}, Point{9, 19}}}, right_result = Leaf{Rect{Point{11, 0}, Point{19, 19}}};
  ASSERT_EQ(*leaf.left, left_result);
  ASSERT_EQ(*leaf.right, right_result);
}

TEST_F(Fixtures, TestBspSplitDebug) {
  // Initialise the resultant 2D grid where the x=10 column is a debug wall
  std::vector<std::vector<TileType>> result_grid(20, std::vector<TileType>(20, TileType::Empty));
  for (int y = 0; y < 20; y++) {
    result_grid[y][10] = TileType::DebugWall;
  }

  // Test the result of 2 debug splits
  leaf.split(grid, random_generator, true);
  leaf.split(grid, random_generator, true);
  ASSERT_EQ(grid, result_grid);
}

TEST_F(Fixtures, TestBspSplitSmallWidthHeight) {
  // Make sure we test what happens if the container's width and height are both
  // less than MIN_CONTAINER_SIZE
  leaf.container = Rect{Point{-1, -1}, Point{-1, -1}};
  ASSERT_FALSE(leaf.split(grid, random_generator, false));
}

TEST_F(Fixtures, TestBspCreateRoomValid) {
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