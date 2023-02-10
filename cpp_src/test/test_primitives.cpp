// External includes
#include "gtest/gtest.h"

// Custom includes
#include "primitives.hpp"
#include "fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestPointSumValid) {
  // Test summing two valid points
  ASSERT_EQ(valid_point_one.sum(valid_point_two), std::make_pair(8, 12));
}

TEST_F(Fixtures, TestPointSumIdentical) {
  // Test summing two identical points
  ASSERT_EQ(valid_point_one.sum(valid_point_one), std::make_pair(6, 10));
}

TEST_F(Fixtures, TestPointSumValidBoundary) {
  // Test summing a valid and boundary point
  ASSERT_EQ(valid_point_one.sum(boundary_point), std::make_pair(7, 5));
}

TEST_F(Fixtures, TestPointSumValidZero) {
  // Test summing a valid and zero point
  ASSERT_EQ(valid_point_one.sum(zero_point), std::make_pair(3, 5));
}

TEST_F(Fixtures, TestPointAbsDiffValid) {
  // Test finding the absolute difference between two valid points
  ASSERT_EQ(valid_point_one.abs_diff(valid_point_two), std::make_pair(2, 2));
}

TEST_F(Fixtures, TestPointAbsDiffIdentical) {
  // Test finding the absolute difference between two identical points
  ASSERT_EQ(valid_point_one.abs_diff(valid_point_one), std::make_pair(0, 0));
}

TEST_F(Fixtures, TestPointAbsDiffValidBoundary) {
  // Test summing a valid and boundary point
  ASSERT_EQ(valid_point_one.abs_diff(boundary_point), std::make_pair(1, 5));
}

TEST_F(Fixtures, TestPointAbsDiffValidZero) {
  // Test finding the absolute difference between a valid and zero point
  ASSERT_EQ(valid_point_one.abs_diff(zero_point), std::make_pair(3, 5));
}

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
