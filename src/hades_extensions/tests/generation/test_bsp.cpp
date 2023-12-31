// External headers
#include <gtest/gtest.h>

// Local headers
#include "generation/bsp.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the generation/bsp.hpp tests.
class BspFixture : public testing::Test {  // NOLINT
 protected:
  /// The random number generator for use in testing.
  std::mt19937 random_generator;

  /// A 2D grid for use in testing.
  const Grid grid{20, 20};

  /// Set up the fixture for the tests.
  void SetUp() override { random_generator.seed(0); }
};

// ----- TESTS ------------------------------
/// Test that the split function correctly splits a leaf vertically.
TEST_F(BspFixture, TestBspSplitVertical) {
  // Split a leaf vertically
  Leaf leaf{{{0, 0}, {15, 10}}};
  split(leaf, random_generator);

  // Make sure the children are correct
  ASSERT_EQ(*leaf.left->container, Rect({0, 0}, {7, 10}));
  ASSERT_EQ(*leaf.right->container, Rect({9, 0}, {15, 10}));
}

/// Test that the split function correctly splits a leaf horizontally.
TEST_F(BspFixture, TestBspSplitHorizontal) {
  // Split a leaf horizontally
  Leaf leaf{{{0, 0}, {10, 15}}};
  split(leaf, random_generator);

  // Make sure the children are correct
  ASSERT_EQ(*leaf.left->container, Rect({0, 0}, {10, 7}));
  ASSERT_EQ(*leaf.right->container, Rect({0, 9}, {10, 15}));
}

/// Test that the split function correctly splits a leaf in a random direction.
TEST_F(BspFixture, TestBspSplitRandom) {
  // Split a leaf randomly
  Leaf leaf{{{0, 0}, {15, 15}}};
  split(leaf, random_generator);

  // Make sure the children are correct
  ASSERT_EQ(*leaf.left->container, Rect({0, 0}, {7, 15}));
  ASSERT_EQ(*leaf.right->container, Rect({9, 0}, {15, 15}));
}

/// Test that the split function correctly splits a leaf multiple times.
TEST_F(BspFixture, TestBspSplitMultiple) {
  // Split a leaf multiple times
  Leaf leaf{{{0, 0}, {20, 20}}};
  split(leaf, random_generator);

  // Make sure the children are correct
  ASSERT_EQ(*leaf.left->container, Rect({0, 0}, {10, 20}));
  ASSERT_EQ(*leaf.left->left->container, Rect({0, 0}, {10, 11}));
  ASSERT_EQ(*leaf.left->right->container, Rect({0, 13}, {10, 20}));
  ASSERT_EQ(*leaf.right->container, Rect({12, 0}, {20, 20}));
  ASSERT_EQ(*leaf.right->left->container, Rect({12, 0}, {20, 10}));
  ASSERT_EQ(*leaf.right->right->container, Rect({12, 12}, {20, 20}));
}

/// Test that the split function returns if the leaf is already split.
TEST_F(BspFixture, TestBspSplitExistingChildren) {
  // Split a leaf that already has children
  Leaf leaf{{{0, 0}, {100, 100}}};
  leaf.left = std::make_unique<Leaf>(Rect{{0, 0}, {0, 0}});
  leaf.right = std::make_unique<Leaf>(Rect{{0, 0}, {0, 0}});
  split(leaf, random_generator);

  // Make sure the children haven't changed
  ASSERT_EQ(*leaf.left->container, Rect({0, 0}, {0, 0}));
  ASSERT_EQ(*leaf.right->container, Rect({0, 0}, {0, 0}));
}

/// Test that the split function overwrites the children if only one child exists.
TEST_F(BspFixture, TestBspSplitSingleChild) {
  // Split a leaf that only has one child
  Leaf leaf{{{0, 0}, {15, 15}}};
  leaf.left = std::make_unique<Leaf>(Rect{{0, 0}, {0, 0}});
  split(leaf, random_generator);

  // Make sure the children are correct
  ASSERT_EQ(*leaf.left->container, Rect({0, 0}, {7, 15}));
  ASSERT_EQ(*leaf.right->container, Rect({9, 0}, {15, 15}));
}

/// Test that the split function returns if the leaf is too small to split.
TEST_F(BspFixture, TestBspSplitTooSmall) {
  Leaf leaf{{{0, 0}, {0, 0}}};
  split(leaf, random_generator);
  ASSERT_EQ(leaf.left, nullptr);
  ASSERT_EQ(leaf.right, nullptr);
}

/// Test that the create_room function creates a room in a leaf correctly.
TEST_F(BspFixture, TestBspCreateRoomSingleLeaf) {
  // Create a room inside a leaf
  std::vector<Rect> rooms;
  Leaf leaf{{{0, 0}, {15, 15}}};
  create_room(leaf, grid, random_generator, rooms);

  // Make sure the room is correct
  ASSERT_EQ(*leaf.room, Rect({4, 4}, {13, 14}));
  ASSERT_EQ(rooms.size(), 1);
}

/// Test that the create_room function creates rooms in the left and right children.
TEST_F(BspFixture, TestBspCreateRoomChildLeafs) {
  // Create a room inside a leaf that has already been split
  std::vector<Rect> rooms;
  Leaf leaf{{{0, 0}, {15, 15}}};
  leaf.left = std::make_unique<Leaf>(Rect{{0, 0}, {7, 15}});
  leaf.right = std::make_unique<Leaf>(Rect{{9, 0}, {15, 15}});
  create_room(leaf, grid, random_generator, rooms);

  // Make sure the room is correct
  ASSERT_EQ(*leaf.left->room, Rect({2, 0}, {6, 6}));
  ASSERT_EQ(*leaf.right->room, Rect({9, 4}, {14, 10}));
  ASSERT_EQ(rooms.size(), 2);
}

/// Test that the create_room function overwrites the room if it already exists.
TEST_F(BspFixture, TestBspCreateRoomExistingRoom) {
  // Create a room inside a leaf that already has a room
  std::vector<Rect> rooms;
  Leaf leaf{{{0, 0}, {15, 15}}};
  leaf.room = std::make_unique<Rect>(Rect{{0, 0}, {0, 0}});
  create_room(leaf, grid, random_generator, rooms);

  // Make sure the room is correct
  ASSERT_EQ(*leaf.room, Rect({4, 4}, {13, 14}));
  ASSERT_EQ(rooms.size(), 1);
}

/// Test that the create_room function throws an exception if the leaf is too small.
TEST_F(BspFixture, TestBspCreateRoomTooSmallLeaf) {
  std::vector<Rect> rooms;
  Leaf leaf{{{0, 0}, {0, 0}}};
  create_room(leaf, grid, random_generator, rooms);
  ASSERT_EQ(leaf.room, nullptr);
  ASSERT_TRUE(rooms.empty());
}
