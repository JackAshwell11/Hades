// External includes
#include "gtest/gtest.h"

// Custom includes
#include "primitives.hpp"
#include "fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestGridConvertPositionMiddle) {
  // Test if a position in the middle can be converted correctly
  ASSERT_EQ(small_grid.convert_position({3, 4}), 27);
}

TEST_F(Fixtures, TestGridConvertPositionEdgeTop) {
  // Test if a position on the top edge can be converted correctly
  ASSERT_EQ(small_grid.convert_position({3, 0}), 3);
}

TEST_F(Fixtures, TestGridConvertPositionEdgeBottom) {
  // Test if a position on the bottom edge can be converted correctly
  ASSERT_EQ(small_grid.convert_position({4, 8}), 52);
}

TEST_F(Fixtures, TestGridConvertPositionEdgeLeft) {
  // Test if a position on the left edge can be converted correctly
  ASSERT_EQ(small_grid.convert_position({0, 7}), 42);
}

TEST_F(Fixtures, TestGridConvertPositionEdgeRight) {
  // Test if a position on the right edge can be converted correctly
  ASSERT_EQ(small_grid.convert_position({2, 8}), 50);
}

TEST_F(Fixtures, TestGridConvertPositionSmall) {
  // Test if converting a position outside the array throws an exception
  ASSERT_THROW(small_grid.convert_position({-1, -1}), std::out_of_range);
}

TEST_F(Fixtures, TestGridConvertPositionLarge) {
  // Test if converting a position outside the array throws an exception
  ASSERT_THROW(small_grid.convert_position({10, 10}), std::out_of_range);
}

TEST_F(Fixtures, TestGridGetValueMiddle) {
  // Test if a position in the middle can be got correctly
  (*small_grid.grid)[47] = TileType::Player;
  ASSERT_EQ(small_grid.get_value({5, 7}), TileType::Player);
}

TEST_F(Fixtures, TestGridGetValueEdge) {
  // Test if a position on the edge can be got correctly
  (*small_grid.grid)[29] = TileType::Player;
  ASSERT_EQ(small_grid.get_value({5, 4}), TileType::Player);
}

TEST_F(Fixtures, TestGridGetValueLarge) {
  // Test if getting a position outside the array throws an exception
  ASSERT_THROW(small_grid.get_value({10, 10}), std::out_of_range);
}

TEST_F(Fixtures, TestGridSetValueMiddle) {
  // Test if a position in the middle can be set correctly
  small_grid.set_value({1, 7}, TileType::Player);
  ASSERT_EQ((*small_grid.grid)[43], TileType::Player);
}

TEST_F(Fixtures, TestGridSetValueEdge) {
  // Test if a position on the edge can be set correctly
  small_grid.set_value({5, 2}, TileType::Player);
  ASSERT_EQ((*small_grid.grid)[17], TileType::Player);
}

TEST_F(Fixtures, TestGridSetValueSmall) {
  // Test if setting a position outside the array throws an exception
  ASSERT_THROW(small_grid.set_value({-1, -1}, TileType::Player), std::out_of_range);
}

TEST_F(Fixtures, TestRectGetDistanceToValid) {
  // Test finding the distance between two valid rects
  ASSERT_EQ(valid_rect_one.get_distance_to(valid_rect_two), 3);
}

TEST_F(Fixtures, TestRectGetDistanceToIdentical) {
  // Test finding the distance between two identical rects
  ASSERT_EQ(valid_rect_one.get_distance_to(valid_rect_one), 0);
}

TEST_F(Fixtures, TestRectGetDistanceToZero) {
  // Test finding the distance between a valid and zero size rect
  ASSERT_EQ(valid_rect_one.get_distance_to(zero_size_rect), 6);
}

TEST_F(Fixtures, TestRectPlaceRectCorrect) {
  // Test if the place_rect function places a rect correctly in the grid
  valid_rect_one.place_rect(small_grid);
  std::vector<TileType> target_result = {
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Wall, TileType::Wall, TileType::Wall,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Wall, TileType::Floor, TileType::Wall,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Wall, TileType::Wall, TileType::Wall,
      TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty,
  };
  ASSERT_EQ(*small_grid.grid, target_result);
}
