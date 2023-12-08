// Local headers
#include "generation/astar.hpp"
#include "macros.hpp"

// ----- FIXTURES ------------------------------
/// Implements the fixture for the generation/astar.hpp tests.
class AstarFixture : public testing::Test {
 protected:
  /// A 2D grid for use in testing.
  Grid grid{6, 9};

  /// A position in the middle of the grid for use in testing.
  const Position position_one{3, 7};

  /// An extra position in the middle of the grid for use in testing.
  const Position position_two{4, 1};

  /// A position on the edge of the grid for use in testing.
  const Position position_three{4, 0};

  /// Add obstacles to the grid for use in testing.
  void add_obstacles() const {
    grid.set_value({1, 3}, TileType::Obstacle);
    grid.set_value({2, 7}, TileType::Obstacle);
    grid.set_value({3, 2}, TileType::Obstacle);
    grid.set_value({3, 3}, TileType::Obstacle);
    grid.set_value({3, 6}, TileType::Obstacle);
    grid.set_value({4, 3}, TileType::Obstacle);
    grid.set_value({4, 6}, TileType::Obstacle);
  }
};

// ----- TESTS ------------------------------
/// Test that A* works in a grid with no obstacles when started in the middle.
TEST_F(AstarFixture, TestCalculateAstarPathNoObstaclesMiddleStart) {
  const std::vector<Position> no_obstacles_result{{4, 1}, {3, 2}, {2, 3}, {3, 4}, {4, 5}, {4, 6}, {3, 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_two), no_obstacles_result);
}

/// Test that A* fails in a grid with no obstacles when ended on the edge.
TEST_F(AstarFixture, TestCalculateAstarPathNoObstaclesBoundaryEnd) {
  const std::vector<Position> no_obstacles_result{{4, 0}, {3, 1}, {3, 2}, {2, 3}, {3, 4}, {4, 5}, {4, 6}, {3, 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_three), no_obstacles_result);
}

/// Test that A* works in a grid with obstacles when started in the middle.
TEST_F(AstarFixture, TestCalculateAstarPathObstaclesMiddleStart) {
  add_obstacles();
  const std::vector<Position> obstacles_result{{4, 1}, {4, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {3, 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_two), obstacles_result);
}

/// Test that A* fails in a grid with obstacles when ended on the edge.
TEST_F(AstarFixture, TestCalculateAstarPathObstaclesBoundaryEnd) {
  add_obstacles();
  const std::vector<Position> obstacles_result{{4, 0}, {3, 1}, {2, 2}, {2, 3}, {3, 4}, {2, 5}, {2, 6}, {3, 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_three), obstacles_result);
}

/// Test that A* fails in an empty grid.
TEST_F(AstarFixture, TestCalculateAstarPathEmptyGrid) {
  const Grid empty_grid{0, 0};
  ASSERT_THROW_MESSAGE(calculate_astar_path(empty_grid, position_one, position_two), std::length_error,
                       "Grid size must be bigger than 0.")
}
