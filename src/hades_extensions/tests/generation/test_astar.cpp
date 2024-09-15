// Local headers
#include "generation/astar.hpp"
#include "generation/primitives.hpp"
#include "macros.hpp"

/// Implements the fixture for the generation/astar.hpp tests.
class AstarFixture : public testing::Test {
 protected:
  /// A 2D grid for use in testing.
  const Grid grid{6, 9};

  /// A position in the middle of the grid for use in testing.
  const Position position_one{.x = 3, .y = 7};

  /// An extra position in the middle of the grid for use in testing.
  const Position position_two{.x = 4, .y = 1};

  /// A position on the edge of the grid for use in testing.
  const Position position_three{.x = 4, .y = 0};

  /// Add obstacles to the grid for use in testing.
  void add_obstacles() const {
    grid.set_value({.x = 1, .y = 3}, TileType::Obstacle);
    grid.set_value({.x = 2, .y = 7}, TileType::Obstacle);
    grid.set_value({.x = 3, .y = 2}, TileType::Obstacle);
    grid.set_value({.x = 3, .y = 3}, TileType::Obstacle);
    grid.set_value({.x = 3, .y = 6}, TileType::Obstacle);
    grid.set_value({.x = 4, .y = 3}, TileType::Obstacle);
    grid.set_value({.x = 4, .y = 6}, TileType::Obstacle);
  }

  /// Add item and floor tiles to the grid for use in testing.
  ///
  /// @param items The positions of the items to add.
  /// @param all Whether to add item tiles to all positions.
  void add_items_and_floors(const std::unordered_set<Position> &items, const bool all = false) const {
    for (int y = 0; y < grid.height; y++) {
      for (int x = 0; x < grid.width; x++) {
        if (all) {
          grid.set_value({.x = x, .y = y}, TileType::Obstacle);
        } else {
          grid.set_value({.x = x, .y = y}, !items.contains({.x = x, .y = y}) ? TileType::Floor : TileType::Obstacle);
        }
      }
    }
  }
};

/// Test that A* works in a grid with no obstacles when started in the middle.
TEST_F(AstarFixture, TestCalculateAstarPathNoObstaclesMiddleStart) {
  const std::vector<Position> no_obstacles_result{{.x = 4, .y = 1}, {.x = 3, .y = 2}, {.x = 2, .y = 3},
                                                  {.x = 3, .y = 4}, {.x = 4, .y = 5}, {.x = 4, .y = 6},
                                                  {.x = 3, .y = 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_two), no_obstacles_result);
}

/// Test that A* works in a grid with no obstacles when ended on the edge.
TEST_F(AstarFixture, TestCalculateAstarPathNoObstaclesBoundaryEnd) {
  const std::vector<Position> no_obstacles_result{{.x = 4, .y = 0}, {.x = 3, .y = 1}, {.x = 3, .y = 2},
                                                  {.x = 2, .y = 3}, {.x = 3, .y = 4}, {.x = 4, .y = 5},
                                                  {.x = 4, .y = 6}, {.x = 3, .y = 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_three), no_obstacles_result);
}

/// Test that A* works in a grid with obstacles when started in the middle.
TEST_F(AstarFixture, TestCalculateAstarPathObstaclesMiddleStart) {
  add_obstacles();
  const std::vector<Position> obstacles_result{{.x = 4, .y = 1}, {.x = 4, .y = 2}, {.x = 5, .y = 3}, {.x = 4, .y = 4},
                                               {.x = 3, .y = 5}, {.x = 2, .y = 6}, {.x = 3, .y = 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_two), obstacles_result);
}

/// Test that A* works in a grid with obstacles when ended on the edge.
TEST_F(AstarFixture, TestCalculateAstarPathObstaclesBoundaryEnd) {
  add_obstacles();
  const std::vector<Position> obstacles_result{{.x = 4, .y = 0}, {.x = 3, .y = 1}, {.x = 2, .y = 2}, {.x = 2, .y = 3},
                                               {.x = 3, .y = 4}, {.x = 2, .y = 5}, {.x = 2, .y = 6}, {.x = 3, .y = 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_three), obstacles_result);
}

/// Test that A* throws an exception in an empty grid.
TEST_F(AstarFixture, TestCalculateAstarPathEmptyGrid) {
  ASSERT_THROW_MESSAGE(calculate_astar_path({0, 0}, position_one, position_two), std::length_error,
                       "Grid size must be bigger than 0.")
}
