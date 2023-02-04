// External includes
#include "gtest/gtest.h"

// Custom includes
#include "astar.hpp"
#include "shared_fixtures.hpp"

// ----- TESTS ------------------------------
TEST_F(Points, TestGridBFS) {
  // Test for a point in the middle of the grid
  std::vector<Point> result_middle = {{2, 4}, {3, 4}, {4, 4}, {2, 5}, {4, 5}, {2, 6}, {3, 6}, {4, 6}};
  ASSERT_EQ(grid_bfs(valid_point_one, 10, 10), result_middle);

  // Test for a point on the edge of the grid
  std::vector<Point> result_edge = {{3, 0}, {5, 0}, {3, 1}, {4, 1}, {5, 1}};
  ASSERT_EQ(grid_bfs(boundary_point, 10, 10), result_edge);
}

TEST_F(Points, TestCalculateAstarPath) {

}
