// External includes
#include "gtest/gtest.h"

// Custom includes
#include "fixtures.hpp"
#include "generation/bsp.hpp"

// ----- TESTS ------------------------------
TEST_F(GenerationFixtures, TestBspSplitCorrect) {
  // Split the bsp normally
  split(leaf, random_generator);

  // Calculate the difference between the bottom right point of the left leaf
  // and the top right point of the right leaf, so we can determine it's split
  // direction
  Point leaf_diff = leaf.left->container->bottom_right - leaf.right->container->top_left;
  Point target_diff = (leaf_diff.x == 2) ? Point{2, 19} : Point{19, 2};

  // Test if the child leafs border each other
  ASSERT_EQ(leaf_diff, target_diff);
}

TEST_F(GenerationFixtures, TestBspSplitSmallWidthHeight) {
  // Make sure we test what happens if the container's width and height are
  // both less than MIN_CONTAINER_SIZE
  leaf.container = std::make_unique<Rect>(Point{-1, -1}, Point{-1, -1});
  ASSERT_FALSE(split(leaf, random_generator));
}

TEST_F(GenerationFixtures, TestBspSplitNotNullLeftRight) {
  // Test what happens if the leaf and right leafs are not null
  leaf.left = std::make_unique<Leaf>(Rect{{0, 0}, {0, 0}});
  leaf.right = std::make_unique<Leaf>(Rect{{0, 0}, {0, 0}});
  ASSERT_FALSE(split(leaf, random_generator));
}

TEST_F(GenerationFixtures, TestBspCreateRoomChildLeaf) {
  // Repeat until a room is created since the ratio may be wrong sometimes then
  // test that the room is not null
  while (!create_room(leaf, grid, random_generator)) {}
  ASSERT_TRUE(leaf.room != nullptr);
}

TEST_F(GenerationFixtures, TestBspCreateRoomNotNullLeftRight) {
  // Test what happens if the leaf and right leafs are not null
  leaf.left = std::make_unique<Leaf>(Rect{{0, 0}, {0, 0}});
  leaf.right = std::make_unique<Leaf>(Rect{{0, 0}, {0, 0}});
  ASSERT_FALSE(create_room(leaf, grid, random_generator));
}
