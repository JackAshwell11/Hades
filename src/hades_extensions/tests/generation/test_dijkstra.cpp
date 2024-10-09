// Std headers
#include <unordered_set>

// Local headers
#include "generation/dijkstra.hpp"
#include "macros.hpp"

/// Implements the fixture for the generation/dijkstra.hpp tests.
class DijkstraFixture : public testing::Test {
 protected:
  /// A 2D grid for use in testing.
  const Grid grid{6, 9};

  /// A position in the middle of the grid for use in testing.
  const cpVect position_one{.x = 3, .y = 7};

  /// An extra position in the middle of the grid for use in testing.
  const cpVect position_two{.x = 4, .y = 1};

  /// A position on the edge of the grid for use in testing.
  const cpVect position_three{.x = 4, .y = 0};

  /// Add obstacles to the grid for use in testing.
  ///
  /// @param all - Whether to add obstacles to the entire grid or not.
  void add_obstacles(const bool all = false) const {
    const std::unordered_set<cpVect> obstacles{{.x = 1, .y = 3}, {.x = 2, .y = 7}, {.x = 3, .y = 2}, {.x = 3, .y = 3},
                                                 {.x = 3, .y = 6}, {.x = 4, .y = 3}, {.x = 4, .y = 6}};
    for (int y{0}; y < grid.height; y++) {
      for (int x{0}; x < grid.width; x++) {
        const cpVect position{cpv(x, y)};
        if (all) {
          grid.set_value(position, TileType::Obstacle);
        } else {
          grid.set_value(position, !obstacles.contains(position) ? TileType::Floor : TileType::Obstacle);
        }
      }
    }
  }

  /// Add floors and walls to the grid for use in testing.
  ///
  /// @param walls - The positions of the walls to add.
  /// @param all - Whether to add walls to the entire grid or not.
  void add_floors(const std::unordered_set<cpVect> &walls, const bool all = false) const {
    for (int y{0}; y < grid.height; y++) {
      for (int x{0}; x < grid.width; x++) {
        if (all) {
          grid.set_value(cpv(x, y), TileType::Wall);
        } else {
          grid.set_value(cpv(x, y), !walls.contains(cpv(x, y)) ? TileType::Floor : TileType::Wall);
        }
      }
    }
  }
};

/// Test that A* works in a grid with no obstacles when started in the middle.
TEST_F(DijkstraFixture, TestCalculateAstarPathNoObstaclesMiddleStart) {
  const std::vector<cpVect> no_obstacles_result{{.x = 4, .y = 1}, {.x = 3, .y = 2}, {.x = 2, .y = 3},
                                                  {.x = 3, .y = 4}, {.x = 4, .y = 5}, {.x = 4, .y = 6},
                                                  {.x = 3, .y = 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_two), no_obstacles_result);
}

/// Test that A* works in a grid with no obstacles when ended on the edge.
TEST_F(DijkstraFixture, TestCalculateAstarPathNoObstaclesBoundaryEnd) {
  const std::vector<cpVect> no_obstacles_result{{.x = 4, .y = 0}, {.x = 3, .y = 1}, {.x = 3, .y = 2},
                                                  {.x = 2, .y = 3}, {.x = 3, .y = 4}, {.x = 4, .y = 5},
                                                  {.x = 4, .y = 6}, {.x = 3, .y = 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_three), no_obstacles_result);
}

/// Test that A* works in a grid with obstacles when started in the middle.
TEST_F(DijkstraFixture, TestCalculateAstarPathObstaclesMiddleStart) {
  add_obstacles();
  const std::vector<cpVect> obstacles_result{{.x = 4, .y = 1}, {.x = 4, .y = 2}, {.x = 5, .y = 3}, {.x = 4, .y = 4},
                                               {.x = 3, .y = 5}, {.x = 2, .y = 6}, {.x = 3, .y = 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_two), obstacles_result);
}

/// Test that A* works in a grid with obstacles when ended on the edge.
TEST_F(DijkstraFixture, TestCalculateAstarPathObstaclesBoundaryEnd) {
  add_obstacles();
  const std::vector<cpVect> obstacles_result{{.x = 4, .y = 0}, {.x = 3, .y = 1}, {.x = 2, .y = 2}, {.x = 2, .y = 3},
                                               {.x = 3, .y = 4}, {.x = 2, .y = 5}, {.x = 2, .y = 6}, {.x = 3, .y = 7}};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_three), obstacles_result);
}

/// Test that A* doesn't work in a grid with only obstacles.
TEST_F(DijkstraFixture, TestCalculateAstarPathOnlyObstacles) {
  add_obstacles(true);
  const std::vector<cpVect> only_walls_result{};
  ASSERT_EQ(calculate_astar_path(grid, position_one, position_two), only_walls_result);
}

/// Test that A* throws an exception in an empty grid.
TEST_F(DijkstraFixture, TestCalculateAstarPathEmptyGrid) {
  ASSERT_EQ(calculate_astar_path({0, 0}, position_one, position_two), std::vector<cpVect>{});
}

/// Test that getting the furthest position works in a grid with no walls.
TEST_F(DijkstraFixture, TestGetFurthestcpVectNoWalls) {
  add_floors({});
  const auto [x, y]{get_furthest_position(grid, position_one)};
  ASSERT_GE(x, 0);
  ASSERT_LE(x, 5);
  ASSERT_EQ(y, 0);
}

/// Test that getting the furthest position works in a grid with walls.
TEST_F(DijkstraFixture, TestGetFurthestcpVectWalls) {
  add_floors({{.x = 2, .y = 4}, {.x = 4, .y = 2}, {.x = 4, .y = 6}});
  constexpr cpVect furthest_position{.x = 4, .y = 0};
  ASSERT_EQ(get_furthest_position(grid, position_one), furthest_position);
}

/// Test that getting the furthest position throws an exception in a grid with only walls.
TEST_F(DijkstraFixture, TestGetFurthestcpVectOnlyWalls) {
  add_floors({}, true);
  constexpr cpVect furthest_position{.x = -1, .y = -1};
  ASSERT_EQ(get_furthest_position(grid, position_one), furthest_position);
}

/// Test that getting the furthest position throws an exception in an empty grid.
TEST_F(DijkstraFixture, TestGetFurthestcpVectEmptyGrid) {
  constexpr cpVect furthest_position{.x = -1, .y = -1};
  ASSERT_EQ(get_furthest_position({0, 0}, position_one), furthest_position);
}
