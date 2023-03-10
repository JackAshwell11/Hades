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
  ASSERT_EQ(small_grid.grid, target_result);
}

TEST_F(Fixtures, TestGridGetValueMiddle) {
  // Test if a position in the middle can be got correctly
  grid.grid[191] = TileType::Player;
  ASSERT_EQ(grid.get_value(11, 9), TileType::Player);
}

TEST_F(Fixtures, TestGridGetValueBoundary) {
  // Test if a position on the edge can be got correctly
  grid.grid[120] = TileType::Player;
  ASSERT_EQ(grid.get_value(0, 6), TileType::Player);
}

TEST_F(Fixtures, TestGridGetValueInvalid) {
  // Test if getting a position outside the array throws an exception
  ASSERT_THROW(grid.get_value(22, 22), std::out_of_range);
}

TEST_F(Fixtures, TestGridSetValueMiddle) {
  // Test if a position in the middle can be set correctly
  grid.set_value(12, 7, TileType::Player);
  ASSERT_EQ(grid.grid[152], TileType::Player);
}

TEST_F(Fixtures, TestGridSetValueBoundary) {
  // Test if a position on the edge can be set correctly
  grid.set_value(20, 5, TileType::Player);
  ASSERT_EQ(grid.grid[120], TileType::Player);
}

TEST_F(Fixtures, TestGridSetValueInvalid) {
  // Test if setting a position outside the array throws an exception
  ASSERT_THROW(grid.set_value(-1, 5, TileType::Player), std::out_of_range);
}

// TODO: DO LOTS MORE TESTS FOR GET/SET VALUE
