// Std includes
#include <stdexcept>

// External includes
#include "gtest/gtest.h"

// Custom includes
#include "astar.hpp"
#include "fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestCalculateAstarPathObstaclesMiddleStart) {
  // Test A* in a grid with obstacles
  std::vector<Point> obstacles_result = {{1, 2}, {2, 3}, {2, 4}, {3, 5}};
  ASSERT_EQ(calculate_astar_path(detailed_grid, valid_point_one, {1, 2}), obstacles_result);
}

TEST_F(Fixtures, TestCalculateAstarPathObstaclesBoundaryStart) {
  // Test A* in a grid with obstacles
  std::vector<Point> obstacles_result = {};
  ASSERT_EQ(calculate_astar_path(detailed_grid, valid_point_one, boundary_point), obstacles_result);
}

TEST_F(Fixtures, TestCalculateAstarPathNoObstaclesMiddleStart) {
  // Test A* in a grid with no obstacles
  std::vector<Point> no_obstacles_result = {{5, 7}, {4, 6}, {3, 5}};
  ASSERT_EQ(calculate_astar_path(grid, valid_point_one, valid_point_two), no_obstacles_result);
}

TEST_F(Fixtures, TestCalculateAstarPathBoundaryGoal) {
  // Test A* with a goal on the boundaries
  std::vector<Point> boundary_result = {};
  ASSERT_EQ(calculate_astar_path(small_grid, valid_point_one, valid_point_two), boundary_result);
}

TEST_F(Fixtures, TestCalculateAstarPathEmptyGrid) {
  // Test A* in an empty grid
  ASSERT_THROW(calculate_astar_path(empty_grid, valid_point_one, boundary_point), std::length_error);
}
