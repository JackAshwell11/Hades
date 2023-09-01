// Std includes
#include <stdexcept>

// External includes
#include "gtest/gtest.h"

// Custom includes
#include "fixtures.hpp"
#include "generation/astar.hpp"

// ----- TESTS ------------------------------
TEST_F(GenerationFixtures, TestCalculateAstarPathObstaclesMiddleStart) {
  // Test A* in a grid with obstacles
  std::vector<Position> obstacles_result = {{1, 2}, {2, 3}, {2, 4}, {3, 5}};
  ASSERT_EQ(calculate_astar_path(detailed_grid, valid_position_one, {1, 2}), obstacles_result);
}

TEST_F(GenerationFixtures, TestCalculateAstarPathObstaclesBoundaryStart) {
  // Test A* in a grid with obstacles
  std::vector<Position> obstacles_result = {};
  ASSERT_EQ(calculate_astar_path(detailed_grid, valid_position_one, boundary_position), obstacles_result);
}

TEST_F(GenerationFixtures, TestCalculateAstarPathNoObstaclesMiddleStart) {
  // Test A* in a grid with no obstacles
  std::vector<Position> no_obstacles_result = {{5, 7}, {4, 6}, {3, 5}};
  ASSERT_EQ(calculate_astar_path(grid, valid_position_one, valid_position_two), no_obstacles_result);
}

TEST_F(GenerationFixtures, TestCalculateAstarPathBoundaryGoal) {
  // Test A* with a goal on the boundaries
  std::vector<Position> boundary_result = {};
  ASSERT_EQ(calculate_astar_path(small_grid, valid_position_one, valid_position_two), boundary_result);
}

TEST_F(GenerationFixtures, TestCalculateAstarPathEmptyGrid) {
  // Test A* in an empty grid
  ASSERT_THROW(calculate_astar_path(empty_grid, valid_position_one, boundary_position), std::length_error);
}
