// Std includes
#include <stdexcept>

// External includes
#include "gtest/gtest.h"

// Custom includes
#include "astar.hpp"
#include "fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Fixtures, TestGridBFSMiddle) {
  // Test for a point in the middle of the grid
  std::vector<Point> result_middle = {{2, 4}, {3, 4}, {4, 4}, {2, 5}, {4, 5}, {2, 6}, {3, 6}, {4, 6}};
  ASSERT_EQ(grid_bfs(valid_point_one, 10, 10), result_middle);
}

TEST_F(Fixtures, TestGridBFSEdge) {
  // Test for a point on the edge of the grid
  std::vector<Point> result_edge = {{3, 0}, {5, 0}, {3, 1}, {4, 1}, {5, 1}};
  ASSERT_EQ(grid_bfs(boundary_point, 10, 10), result_edge);
}

TEST_F(Fixtures, TestCalculateAstarPathNoObstacles) {
  // Test A* in a grid with no obstacles
  // TODO: FIX RESULT
  std::vector<Point> no_obstacles_result = {{4, 0}, {4, 1}, {4, 2}, {5, 3}, {4, 4}, {3, 5}};
  ASSERT_EQ(calculate_astar_path(grid, valid_point_one, boundary_point), no_obstacles_result);
}

TEST_F(Fixtures, TestCalculateAstarPathObstacles) {
  // Test A* in a grid with obstacles
  std::vector<Point> obstacles_result = {{4, 0}, {3, 1}, {4, 2}, {5, 3}, {4, 4}, {3, 5}};
  ASSERT_EQ(calculate_astar_path(detailed_grid, valid_point_one, boundary_point), obstacles_result);
}

TEST_F(Fixtures, TestCalculateAstarPathEmpty) {
  // Test A* in an empty grid
  ASSERT_THROW(calculate_astar_path(empty_grid, valid_point_one, boundary_point), std::length_error);
}
