// External includes
#include "gtest/gtest.h"

// Custom includes
#include "primitives.hpp"
#include "shared_fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Points, TestPointSum) {
  // Test summing two valid points
  ASSERT_EQ(valid_point_one.sum(valid_point_two), std::make_pair(8, 12));

  // Test summing two identical points
  ASSERT_EQ(valid_point_one.sum(valid_point_one), std::make_pair(6, 10));

  // Test summing a valid and boundary point
  ASSERT_EQ(valid_point_one.sum(boundary_point), std::make_pair(7, 5));

  // Test summing a valid and zero point
  ASSERT_EQ(valid_point_one.sum(zero_point), std::make_pair(3, 5));
}

TEST_F(Points, TestPointAbsDiff) {
  // Test finding the absolute difference between two valid points
  ASSERT_EQ(valid_point_one.abs_diff(valid_point_two), std::make_pair(2, 2));

  // Test finding the absolute difference between two identical points
  ASSERT_EQ(valid_point_one.abs_diff(valid_point_one), std::make_pair(0, 0));

  // Test summing a valid and boundary point
  ASSERT_EQ(valid_point_one.abs_diff(boundary_point), std::make_pair(1, 5));

  // Test finding the absolute difference between a valid and zero point
  ASSERT_EQ(valid_point_one.abs_diff(zero_point), std::make_pair(3, 5));
}

TEST_F(Points, TestRectGetDistanceTo) {
  // Create some rects
  Rect validRectOne{valid_point_one, valid_point_two}, validRectTwo{valid_point_one, boundary_point},
      zeroSizeRect{zero_point, zero_point};

  // Test finding the distance between two valid rects
  ASSERT_EQ(validRectOne.get_distance_to(validRectTwo), 3);

  // Test finding the distance between two identical rects
  ASSERT_EQ(validRectOne.get_distance_to(validRectOne), 0);

  // Test finding the distance between a valid and zero size rect
  ASSERT_EQ(validRectOne.get_distance_to(zeroSizeRect), 6);
}

TEST_F(Points, TestRectPlaceRect) {
  // Create a 2D vector and place a rect inside of it
  std::vector<std::vector<TileType>> grid(9, std::vector<TileType>(6, TileType::Empty));
  Rect{valid_point_one, valid_point_two}.place_rect(grid);

  // Check if the place_rect function places a rect correctly inside a 2D vector
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
  ASSERT_EQ(grid, target_result);
}
