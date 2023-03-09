// External includes
#include "gtest/gtest.h"

// Custom includes
#include "primitives.hpp"
#include "fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestRectGetDistanceToValid) {
  // Test finding the distance between two valid rects
  ASSERT_EQ(valid_rect_one.get_distance_to(valid_rect_two), 3);
}

TEST_F(Fixtures, TestRectGetDistanceToIdentical) {
  // Test finding the distance between two identical rects
  ASSERT_EQ(valid_rect_one.get_distance_to(valid_rect_one), 0);
}

TEST_F(Fixtures, TestRectGetDistanceToValidZero) {
  // Test finding the distance between a valid and zero size rect
  ASSERT_EQ(valid_rect_one.get_distance_to(zero_size_rect), 6);
}

TEST_F(Fixtures, TestRectPlaceRect) {
  // Test if the place_rect function places a rect correctly in the grid
  valid_rect_one.place_rect(small_grid);
  std::vector<std::vector<TileType>> target_result = {
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Wall, TileType::Wall, TileType::Wall},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Wall, TileType::Floor, TileType::Wall},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Wall, TileType::Wall, TileType::Wall},
      {TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty, TileType::Empty},
  };
  ASSERT_EQ(small_grid, target_result);
}
