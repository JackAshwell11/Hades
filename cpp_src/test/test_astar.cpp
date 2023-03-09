// Std includes
#include <stdexcept>

// External includes
#include "gtest/gtest.h"

// Custom includes
#include "astar.hpp"
#include "fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestCalculateAstarPathObstaclesValid) {
  // Test A* in a grid with obstacles
  std::vector<Point> obstacles_result = {{5, 7}, {5, 6}, {4, 5}, {3, 5}};
  ASSERT_EQ(calculate_astar_path(detailed_grid, valid_point_one, valid_point_two), obstacles_result);
}

TEST_F(Fixtures, TestCalculateAstarPathObstaclesBoundary) {
  // Test A* in a grid with obstacles
  std::vector<Point> obstacles_result = {{4, 0}, {3, 1}, {2, 2}, {2, 3}, {2, 4}, {3, 5}};
  ASSERT_EQ(calculate_astar_path(detailed_grid, valid_point_one, boundary_point), obstacles_result);
}

TEST_F(Fixtures, TestCalculateAstarPathNoObstacles) {
  // Test A* in a grid with no obstacles
  std::vector<Point> no_obstacles_result = {{5, 7}, {4, 6}, {3, 5}};
  ASSERT_EQ(calculate_astar_path(grid, valid_point_one, valid_point_two), no_obstacles_result);
}

TEST_F(Fixtures, TestCalculateAstarPathEmpty) {
  // Test A* in an empty grid
  ASSERT_THROW(calculate_astar_path(empty_grid, valid_point_one, boundary_point), std::length_error);
}
